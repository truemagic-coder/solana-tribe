from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, HttpUrl, EmailStr
from typing import Optional, List, Dict, Any, Union
import uvicorn
from datetime import datetime
import httpx
import base64
import json
from starlette.middleware.base import BaseHTTPMiddleware
import os
from httpsig import HeaderSigner, verify_headers
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from dotenv import load_dotenv
from taskiq_redis import ListQueueBroker
from taskiq import SimpleRetryMiddleware
from rdflib import Graph, Namespace
from rdflib.plugin import register, Parser, Serializer

load_dotenv()

# Define ActivityStreams namespace
AS = Namespace("https://www.w3.org/ns/activitystreams#")

app = FastAPI()

SERVER_PRIVATE_KEY = os.getenv("SERVER_PRIVATE_KEY")

if not SERVER_PRIVATE_KEY:
    raise ValueError("SERVER_PRIVATE_KEY environment variable is not set")

# Convert the string representation of the key to a proper key object
server_private_key = serialization.load_pem_private_key(
    SERVER_PRIVATE_KEY.encode(), password=None
)

# Register JSON-LD plugin
register("json-ld", Parser, "rdflib_jsonld.parser", "JsonLDParser")
register("json-ld", Serializer, "rdflib_jsonld.serializer", "JsonLDSerializer")

broker = ListQueueBroker(os.getenv("REDIS_URL")).with_middlewares(
    SimpleRetryMiddleware(default_retry_count=3),
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

client = AsyncIOMotorClient(os.getenv("MONGO_URL"))
db = client["activitypub_db"]

BASE_URL = "https://example.com"


@broker.task
async def deliver_activity_to_follower(activity_dict: dict, follower: str):
    from main import create_signed_headers  # Import here to avoid circular imports
    from httpx import AsyncClient

    async with AsyncClient() as client:
        headers = await create_signed_headers("POST", follower, activity_dict)
        try:
            response = await client.post(follower, json=activity_dict, headers=headers)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to deliver activity to {follower}: {e}")


class PublicKey(BaseModel):
    id: str
    owner: str
    publicKeyPem: str


class Actor(BaseModel):
    context: List[str] = Field(
        default=[
            "https://www.w3.org/ns/activitystreams",
            "https://w3id.org/security/v1",
        ],
        alias="@context",
    )
    type: str
    id: HttpUrl
    inbox: HttpUrl
    outbox: HttpUrl
    followers: HttpUrl
    following: HttpUrl
    liked: HttpUrl
    streams: List[HttpUrl]
    preferredUsername: str
    name: Optional[str]
    summary: Optional[str]
    publicKey: PublicKey
    icon: Optional[HttpUrl]
    image: Optional[HttpUrl]
    endpoints: Dict[str, HttpUrl]


class Activity(BaseModel):
    context: List[str] = Field(
        default=["https://www.w3.org/ns/activitystreams"], alias="@context"
    )
    type: str
    id: Optional[HttpUrl]
    actor: Union[HttpUrl, Actor]
    object: Optional[Union[Dict[str, Any], HttpUrl, "Activity"]]
    target: Optional[Union[Dict[str, Any], HttpUrl, "Activity"]]
    result: Optional[Union[Dict[str, Any], HttpUrl, "Activity"]]
    origin: Optional[Union[Dict[str, Any], HttpUrl, "Activity"]]
    instrument: Optional[Union[Dict[str, Any], HttpUrl, "Activity"]]
    to: Optional[List[Union[HttpUrl, Actor]]] = []
    bto: Optional[List[Union[HttpUrl, Actor]]] = []
    cc: Optional[List[Union[HttpUrl, Actor]]] = []
    bcc: Optional[List[Union[HttpUrl, Actor]]] = []
    audience: Optional[List[Union[HttpUrl, Actor]]] = []
    published: Optional[datetime]
    updated: Optional[datetime]


class OrderedCollection(BaseModel):
    context: List[str] = Field(
        default=["https://www.w3.org/ns/activitystreams"], alias="@context"
    )
    type: str = "OrderedCollection"
    totalItems: int
    first: HttpUrl
    last: HttpUrl


class OrderedCollectionPage(BaseModel):
    context: List[str] = Field(
        default=["https://www.w3.org/ns/activitystreams"], alias="@context"
    )
    type: str = "OrderedCollectionPage"
    id: HttpUrl
    totalItems: int
    orderedItems: List[Activity]
    next: Optional[HttpUrl]
    prev: Optional[HttpUrl]


class ContentNegotiationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        accept = request.headers.get("Accept", "application/activity+json")
        response = await call_next(request)

        if "application/ld+json" in accept:
            response.headers["Content-Type"] = "application/ld+json"
            body = await response.body()
            json_body = json.loads(body)

            # Convert to RDF graph
            g = Graph()
            g.parse(data=json.dumps(json_body), format="json-ld")

            # Define ActivityStreams context
            context = {
                "@context": {
                    "as": "https://www.w3.org/ns/activitystreams#",
                    "id": "@id",
                    "type": "@type",
                }
            }

            # Serialize back to JSON-LD, using ActivityStreams context
            ld_body = g.serialize(format="json-ld", context=context, auto_compact=True)

            return Response(content=ld_body, media_type="application/ld+json")

        return response


app.add_middleware(ContentNegotiationMiddleware)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    # In a real implementation, you'd validate the token properly
    if token != "fake-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    return {"username": "testuser"}


@app.get("/.well-known/webfinger")
async def webfinger(resource: str):
    username = resource.split("acct:")[1].split("@")[0]
    actor = await db.actors.find_one({"preferredUsername": username})
    if not actor:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "subject": f"acct:{username}@example.com",
        "links": [
            {
                "rel": "self",
                "type": "application/activity+json",
                "href": f"{BASE_URL}/actors/{username}",
            }
        ],
    }


@app.get("/actors/{username}", response_model=Actor)
async def get_actor(username: str):
    actor = await db.actors.find_one({"preferredUsername": username})
    if actor:
        return Actor(**actor)
    raise HTTPException(status_code=404, detail="Actor not found")


@app.get("/actors/{username}/inbox", response_model=OrderedCollection)
async def get_inbox(username: str, current_user: dict = Depends(get_current_user)):
    if current_user["username"] != username:
        raise HTTPException(status_code=403, detail="Forbidden")
    count = await db.inboxes.count_documents({"username": username})
    return OrderedCollection(
        totalItems=count,
        first=f"{BASE_URL}/actors/{username}/inbox?page=1",
        last=f"{BASE_URL}/actors/{username}/inbox?page={(count-1)//20 + 1}",
    )


@app.get("/actors/{username}/inbox", response_model=OrderedCollectionPage)
async def get_inbox_page(
    username: str, page: int = 1, current_user: dict = Depends(get_current_user)
):
    if current_user["username"] != username:
        raise HTTPException(status_code=403, detail="Forbidden")
    items = (
        await db.inboxes.find({"username": username})
        .skip((page - 1) * 20)
        .limit(20)
        .to_list(20)
    )
    total = await db.inboxes.count_documents({"username": username})
    return OrderedCollectionPage(
        id=f"{BASE_URL}/actors/{username}/inbox?page={page}",
        totalItems=total,
        orderedItems=items,
        next=f"{BASE_URL}/actors/{username}/inbox?page={page+1}"
        if page * 20 < total
        else None,
        prev=f"{BASE_URL}/actors/{username}/inbox?page={page-1}" if page > 1 else None,
    )


@app.post("/actors/{username}/inbox")
async def post_inbox(username: str, activity: Activity, request: Request):
    verify_http_signature(request)
    await db.inboxes.insert_one(activity.dict())
    await process_activity(activity)
    return Response(status_code=202)


@app.get("/actors/{username}/outbox", response_model=OrderedCollection)
async def get_outbox(username: str):
    count = await db.outboxes.count_documents({"username": username})
    return OrderedCollection(
        totalItems=count,
        first=f"{BASE_URL}/actors/{username}/outbox?page=1",
        last=f"{BASE_URL}/actors/{username}/outbox?page={(count-1)//20 + 1}",
    )


@app.get("/actors/{username}/outbox", response_model=OrderedCollectionPage)
async def get_outbox_page(username: str, page: int = 1):
    items = (
        await db.outboxes.find({"username": username})
        .skip((page - 1) * 20)
        .limit(20)
        .to_list(20)
    )
    total = await db.outboxes.count_documents({"username": username})
    return OrderedCollectionPage(
        id=f"{BASE_URL}/actors/{username}/outbox?page={page}",
        totalItems=total,
        orderedItems=items,
        next=f"{BASE_URL}/actors/{username}/outbox?page={page+1}"
        if page * 20 < total
        else None,
        prev=f"{BASE_URL}/actors/{username}/outbox?page={page-1}" if page > 1 else None,
    )


@app.post("/actors/{username}/outbox")
async def post_outbox(
    username: str, activity: Activity, current_user: dict = Depends(get_current_user)
):
    if current_user["username"] != username:
        raise HTTPException(status_code=403, detail="Forbidden")
    activity.id = activity.id or f"{BASE_URL}/activities/{username}/{id(activity)}"
    activity.published = datetime.utcnow()
    await db.outboxes.insert_one(activity.dict())
    # Get followers
    followers = await db.followers.find_one({"username": username})
    if followers:
        await forward_to_followers(activity.dict(), followers.get("items", []))

    return JSONResponse(status_code=201, content={"id": activity.id})


@app.get("/actors/{username}/followers", response_model=OrderedCollection)
async def get_followers(username: str):
    count = await db.followers.count_documents({"username": username})
    return OrderedCollection(
        totalItems=count,
        first=f"{BASE_URL}/actors/{username}/followers?page=1",
        last=f"{BASE_URL}/actors/{username}/followers?page={(count-1)//20 + 1}",
    )


@app.get("/actors/{username}/followers", response_model=OrderedCollectionPage)
async def get_followers_page(username: str, page: int = 1):
    items = (
        await db.followers.find({"username": username})
        .skip((page - 1) * 20)
        .limit(20)
        .to_list(20)
    )
    total = await db.followers.count_documents({"username": username})
    return OrderedCollectionPage(
        id=f"{BASE_URL}/actors/{username}/followers?page={page}",
        totalItems=total,
        orderedItems=items,
        next=f"{BASE_URL}/actors/{username}/followers?page={page+1}"
        if page * 20 < total
        else None,
        prev=f"{BASE_URL}/actors/{username}/followers?page={page-1}"
        if page > 1
        else None,
    )


@app.get("/actors/{username}/following", response_model=OrderedCollection)
async def get_following(username: str):
    count = await db.following.count_documents({"username": username})
    return OrderedCollection(
        totalItems=count,
        first=f"{BASE_URL}/actors/{username}/following?page=1",
        last=f"{BASE_URL}/actors/{username}/following?page={(count-1)//20 + 1}",
    )


@app.get("/actors/{username}/following", response_model=OrderedCollectionPage)
async def get_following_page(username: str, page: int = 1):
    items = (
        await db.following.find({"username": username})
        .skip((page - 1) * 20)
        .limit(20)
        .to_list(20)
    )
    total = await db.following.count_documents({"username": username})
    return OrderedCollectionPage(
        id=f"{BASE_URL}/actors/{username}/following?page={page}",
        totalItems=total,
        orderedItems=items,
        next=f"{BASE_URL}/actors/{username}/following?page={page+1}"
        if page * 20 < total
        else None,
        prev=f"{BASE_URL}/actors/{username}/following?page={page-1}"
        if page > 1
        else None,
    )


async def deliver_activity(activity: Activity):
    recipients = set(
        activity.to + activity.cc + activity.bto + activity.bcc + activity.audience
    )
    for recipient in recipients:
        if isinstance(recipient, str):
            inbox = recipient
        else:
            inbox = recipient.inbox
        try:
            async with httpx.AsyncClient() as client:
                headers = await create_signed_headers("POST", inbox, activity.dict())
                response = await client.post(
                    str(inbox), json=activity.dict(), headers=headers
                )
                response.raise_for_status()
        except httpx.HTTPError as e:
            print(f"Failed to deliver activity to {inbox}: {e}")


async def process_activity(activity: Activity):
    if activity.type == "Follow":
        await handle_follow(activity)
    elif activity.type == "Undo" and activity.object.get("type") == "Follow":
        await handle_unfollow(activity)
    elif activity.type in ["Create", "Update", "Delete", "Like", "Announce"]:
        await handle_object_activity(activity)


async def handle_follow(follow_activity: Activity):
    follower = follow_activity.actor
    followed = follow_activity.object
    await db.followers.update_one(
        {"username": followed}, {"$addToSet": {"items": follower}}, upsert=True
    )
    accept_activity = Activity(
        type="Accept", actor=followed, object=follow_activity.dict(), to=[follower]
    )
    await deliver_activity(accept_activity)


async def handle_unfollow(unfollow_activity: Activity):
    follower = unfollow_activity.actor
    followed = unfollow_activity.object
    await db.followers.update_one(
        {"username": followed}, {"$pull": {"items": follower}}
    )


async def handle_object_activity(activity: Activity):
    if activity.type == "Create":
        await handle_create(activity)
    elif activity.type == "Update":
        await handle_update(activity)
    elif activity.type == "Delete":
        await handle_delete(activity)
    elif activity.type == "Like":
        await handle_like(activity)
    elif activity.type == "Announce":
        await handle_announce(activity)


async def handle_create(activity: Activity):
    object_data = activity.object
    if isinstance(object_data, dict):
        object_id = object_data.get("id") or f"{BASE_URL}/objects/{id(object_data)}"
        object_data["id"] = object_id
        await db.objects.insert_one(object_data)
    await forward_to_followers(activity)


async def handle_update(activity: Activity):
    object_data = activity.object
    if isinstance(object_data, dict):
        object_id = object_data.get("id")
        if object_id:
            await db.objects.update_one({"id": object_id}, {"$set": object_data})
    await forward_to_followers(activity)


async def handle_delete(activity: Activity):
    object_id = activity.object
    if isinstance(object_id, str):
        await db.objects.delete_one({"id": object_id})
    await forward_to_followers(activity)


async def handle_like(activity: Activity):
    liked_object = activity.object
    if isinstance(liked_object, str):
        await db.liked.update_one(
            {"username": activity.actor},
            {"$addToSet": {"items": liked_object}},
            upsert=True,
        )
    await forward_to_followers(activity)


async def handle_announce(activity: Activity):
    announced_object = activity.object
    if isinstance(announced_object, str):
        await db.announced.update_one(
            {"username": activity.actor},
            {"$addToSet": {"items": announced_object}},
            upsert=True,
        )
    await forward_to_followers(activity)


async def forward_to_followers(activity: Activity):
    actor_username = (
        activity.actor.split("/")[-1]
        if isinstance(activity.actor, str)
        else activity.actor.preferredUsername
    )
    followers = await db.followers.find_one({"username": actor_username})
    if followers:
        for follower in followers.get("items", []):
            await deliver_activity(activity, [follower])


async def verify_http_signature(request: Request):
    signature_header = request.headers.get("Signature")
    if not signature_header:
        raise HTTPException(status_code=401, detail="Missing Signature header")

    # Parse the Signature header to get the key_id
    sig_parts = dict(part.split("=", 1) for part in signature_header.split(","))
    key_id = sig_parts.get("keyId")

    if not key_id:
        raise HTTPException(status_code=401, detail="Invalid Signature header")

    # Fetch the public key
    actor_url = key_id.split("#")[0]
    async with httpx.AsyncClient() as client:
        response = await client.get(
            actor_url, headers={"Accept": "application/activity+json"}
        )
        actor_data = response.json()

    public_key_pem = actor_data.get("publicKey", {}).get("publicKeyPem")
    if not public_key_pem:
        raise HTTPException(status_code=401, detail="Unable to fetch public key")

    public_key = serialization.load_pem_public_key(public_key_pem.encode())

    # Verify the signature
    try:
        verify_headers(
            public_key,
            headers=dict(request.headers),
            method=request.method,
            path=request.url.path,
            required_headers=["(request-target)", "host", "date", "digest"],
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid signature: {str(e)}")


def create_signed_headers(method: str, url: str, body: dict):
    digest = base64.b64encode(
        hashes.SHA256(json.dumps(body).encode()).digest()
    ).decode()

    headers = {
        "Host": BASE_URL,
        "Date": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
        "Digest": f"SHA-256={digest}",
        "Content-Type": "application/activity+json",
    }

    signer = HeaderSigner(
        key_id=f"{BASE_URL}/actors/server#main-key",
        private_key=server_private_key,
        headers=["(request-target)", "host", "date", "digest"],
    )

    signed_headers = signer.sign(headers, method=method.lower(), path=url)

    return signed_headers


@app.post("/users", status_code=201)
async def create_user(username: str, email: EmailStr):
    private_key, public_key = generate_key_pair()
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()

    actor = Actor(
        type="Person",
        id=f"{BASE_URL}/actors/{username}",
        inbox=f"{BASE_URL}/actors/{username}/inbox",
        outbox=f"{BASE_URL}/actors/{username}/outbox",
        followers=f"{BASE_URL}/actors/{username}/followers",
        following=f"{BASE_URL}/actors/{username}/following",
        liked=f"{BASE_URL}/actors/{username}/liked",
        streams=[],
        preferredUsername=username,
        name=username,
        summary=f"This is {username}'s account",
        publicKey=PublicKey(
            id=f"{BASE_URL}/actors/{username}#main-key",
            owner=f"{BASE_URL}/actors/{username}",
            publicKeyPem=public_key_pem,
        ),
        endpoints={"sharedInbox": f"{BASE_URL}/shared_inbox"},
    )

    await db.actors.insert_one(actor.dict())
    await db.private_keys.insert_one(
        {
            "username": username,
            "private_key": private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ).decode(),
        }
    )

    return {"message": "User created successfully"}


def generate_key_pair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key


@app.post("/shared_inbox")
async def shared_inbox(activity: Activity, request: Request):
    verify_http_signature(request)
    # Process the activity for all local users
    local_users = await db.actors.find().to_list(None)
    for user in local_users:
        await db.inboxes.insert_one(
            {**activity.dict(), "username": user["preferredUsername"]}
        )
        await process_activity(activity)
    return Response(status_code=202)


@app.get("/actors/{username}/liked", response_model=OrderedCollection)
async def get_liked(username: str):
    count = await db.liked.count_documents({"username": username})
    return OrderedCollection(
        totalItems=count,
        first=f"{BASE_URL}/actors/{username}/liked?page=1",
        last=f"{BASE_URL}/actors/{username}/liked?page={(count-1)//20 + 1}",
    )


@app.get("/actors/{username}/liked", response_model=OrderedCollectionPage)
async def get_liked_page(username: str, page: int = 1):
    items = (
        await db.liked.find({"username": username})
        .skip((page - 1) * 20)
        .limit(20)
        .to_list(20)
    )
    total = await db.liked.count_documents({"username": username})
    return OrderedCollectionPage(
        id=f"{BASE_URL}/actors/{username}/liked?page={page}",
        totalItems=total,
        orderedItems=items,
        next=f"{BASE_URL}/actors/{username}/liked?page={page+1}"
        if page * 20 < total
        else None,
        prev=f"{BASE_URL}/actors/{username}/liked?page={page-1}" if page > 1 else None,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
