from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, HttpUrl, EmailStr
from typing import Optional, List, Dict, Any, Union
import uvicorn
from datetime import datetime, timedelta
import httpx
import base64
import json
from starlette.middleware.base import BaseHTTPMiddleware
import os
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from dotenv import load_dotenv
from taskiq_redis import ListQueueBroker
from taskiq import SimpleRetryMiddleware
from rdflib import Graph, Namespace
from rdflib.plugin import register, Parser, Serializer
import jwt
from passlib.context import CryptContext


load_dotenv()

def load_key(env_var_name, is_pem=True, is_private=True):
    key_data = os.getenv(env_var_name)
    if not is_pem:
        return key_data
    if not key_data:
        raise ValueError(f"{env_var_name} environment variable is not set")
    
    # Check if the key_data is a file path
    if os.path.isfile(key_data):
        with open(key_data, 'rb') as key_file:
            key_data = key_file.read()
    else:
        key_data = key_data.encode()
    
    try:
        if is_private:
            return serialization.load_pem_private_key(key_data, password=None)
        else:
            return serialization.load_pem_public_key(key_data)
    except ValueError:
        # If it's not a PEM, it might be a plain secret key
        return key_data.decode()

# Define ActivityStreams namespace
AS = Namespace("https://www.w3.org/ns/activitystreams#")

app = FastAPI()

SERVER_PRIVATE_KEY = load_key("SERVER_PRIVATE_KEY")
HS256_SECRET_KEY = load_key("HS256_SECRET_KEY", is_pem=False)
RS256_PUBLIC_KEY = load_key("RS256_PUBLIC_KEY", is_private=False)
RS256_PRIVATE_KEY = load_key("RS256_PRIVATE_KEY")

ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Register JSON-LD plugin
register("json-ld", Parser, "rdflib_jsonld.parser", "JsonLDParser")
register("json-ld", Serializer, "rdflib_jsonld.serializer", "JsonLDSerializer")

REDIS_URL = os.getenv("REDIS_URL")

if not REDIS_URL:
    raise ValueError("REDIS_URL environment variable is not set")

broker = ListQueueBroker(REDIS_URL).with_middlewares(
    SimpleRetryMiddleware(default_retry_count=3),
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    raise ValueError("MONGO_URL environment variable is not set")

client = AsyncIOMotorClient(MONGO_URL)
db = client["activitypub_db"]

BASE_URL = os.getenv("BASE_URL")

if not BASE_URL:
    raise ValueError("BASE_URL environment variable is not set")


@broker.task(retry_on_error=True, max_retries=20)
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


class Source(BaseModel):
    content: str
    mediaType: str


class Activity(BaseModel):
    context: List[str] = Field(
        default=["https://www.w3.org/ns/activitystreams"], alias="@context"
    )
    type: str
    id: Optional[HttpUrl]
    actor: Union[HttpUrl, "Actor"]
    object: Optional[Union[Dict[str, Any], HttpUrl, "Activity"]]
    target: Optional[Union[Dict[str, Any], HttpUrl, "Activity"]]
    result: Optional[Union[Dict[str, Any], HttpUrl, "Activity"]]
    origin: Optional[Union[Dict[str, Any], HttpUrl, "Activity"]]
    instrument: Optional[Union[Dict[str, Any], HttpUrl, "Activity"]]
    to: Optional[List[Union[HttpUrl, "Actor"]]] = []
    bto: Optional[List[Union[HttpUrl, "Actor"]]] = []
    cc: Optional[List[Union[HttpUrl, "Actor"]]] = []
    bcc: Optional[List[Union[HttpUrl, "Actor"]]] = []
    audience: Optional[List[Union[HttpUrl, "Actor"]]] = []
    published: Optional[datetime]
    updated: Optional[datetime]
    source: Optional[Source]


# New: Tombstone model
class Tombstone(BaseModel):
    type: str = "Tombstone"
    id: HttpUrl
    formerType: Optional[str]
    deleted: datetime


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

        if "application/ld+json" in accept or "application/activity+json" in accept:
            response.headers["Content-Type"] = (
                'application/ld+json; profile="https://www.w3.org/ns/activitystreams"'
            )
            body = await response.body()
            json_body = json.loads(body)

            # Convert to RDF graph
            g = Graph()
            g.parse(data=json.dumps(json_body), format="json-ld")

            # Define ActivityStreams context
            context = {
                "@context": [
                    "https://www.w3.org/ns/activitystreams",
                    {"@language": "en"},
                ]
            }

            # Serialize back to JSON-LD, using ActivityStreams context
            ld_body = g.serialize(format="json-ld", context=context, auto_compact=True)

            return Response(
                content=ld_body,
                media_type='application/ld+json; profile="https://www.w3.org/ns/activitystreams"',
            )

        return response


app.add_middleware(ContentNegotiationMiddleware)


# Error handling for unsupported methods
@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE"],
)
async def catch_all(request: Request, path: str):
    if request.method not in ["GET", "POST"]:
        raise HTTPException(status_code=405, detail="Method Not Allowed")
    raise HTTPException(status_code=404, detail="Not Found")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


async def authenticate_user(username: str, password: str):
    user = await db.users.find_one({"username": username})
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(
    data: dict, expires_delta: Optional[timedelta] = None, algorithm: str = ALGORITHM
):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})

    if algorithm == "HS256":
        return jwt.encode(to_encode, HS256_SECRET_KEY, algorithm=algorithm)
    elif algorithm == "RS256":
        private_key = serialization.load_pem_private_key(
            RS256_PRIVATE_KEY.encode(), password=None
        )
        return jwt.encode(to_encode, private_key, algorithm=algorithm)
    else:
        raise ValueError("Unsupported algorithm")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Try HS256 first
        try:
            payload = jwt.decode(token, HS256_SECRET_KEY, algorithms=["HS256"])
        except jwt.InvalidTokenError:
            # If HS256 fails, try RS256
            public_key = serialization.load_pem_public_key(RS256_PUBLIC_KEY.encode())
            payload = jwt.decode(token, public_key, algorithms=["RS256"])

        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = await db.users.find_one({"username": username})
    if user is None:
        raise credentials_exception
    return user


@app.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), algorithm: str = ALGORITHM
):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires,
        algorithm=algorithm,
    )
    return {"access_token": access_token, "token_type": "bearer"}


# New: Implement shares collection
@app.get("/objects/{object_id}/shares", response_model=OrderedCollection)
async def get_shares(object_id: str):
    count = await db.shares.count_documents({"object": object_id})
    return OrderedCollection(
        totalItems=count,
        first=f"{BASE_URL}/objects/{object_id}/shares?page=1",
        last=f"{BASE_URL}/objects/{object_id}/shares?page={(count-1)//20 + 1}",
    )


@app.get("/objects/{object_id}/shares", response_model=OrderedCollectionPage)
async def get_shares_page(object_id: str, page: int = 1):
    items = (
        await db.shares.find({"object": object_id})
        .skip((page - 1) * 20)
        .limit(20)
        .to_list(20)
    )
    total = await db.shares.count_documents({"object": object_id})
    return OrderedCollectionPage(
        id=f"{BASE_URL}/objects/{object_id}/shares?page={page}",
        totalItems=total,
        orderedItems=items,
        next=f"{BASE_URL}/objects/{object_id}/shares?page={page+1}"
        if page * 20 < total
        else None,
        prev=f"{BASE_URL}/objects/{object_id}/shares?page={page-1}"
        if page > 1
        else None,
    )


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

    # De-duplicate activities
    existing_activity = await db.inboxes.find_one(
        {"username": username, "id": activity.id}
    )

    if not existing_activity:
        await db.inboxes.insert_one({**activity.dict(), "username": username})
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
                headers = await create_signature("POST", inbox, activity.dict())
                response = await client.post(
                    str(inbox), json=activity.dict(), headers=headers
                )
                response.raise_for_status()
        except httpx.HTTPError as e:
            print(f"Failed to deliver activity to {inbox}: {e}")


async def process_activity(activity: Activity):
    if activity.type == "Create":
        await handle_create(activity)
    elif activity.type == "Update":
        await handle_update(activity)
    elif activity.type == "Delete":
        await handle_delete(activity)
    elif activity.type == "Follow":
        await handle_follow(activity)
    elif activity.type == "Accept":
        await handle_accept(activity)
    elif activity.type == "Reject":
        await handle_reject(activity)
    elif activity.type == "Add":
        await handle_add(activity)
    elif activity.type == "Remove":
        await handle_remove(activity)
    elif activity.type == "Like":
        await handle_like(activity)
    elif activity.type == "Announce":
        await handle_announce(activity)
    elif activity.type == "Undo":
        await handle_undo(activity)
    elif activity.type == "Block":
        await handle_block(activity)


async def handle_add(activity: Activity):
    if isinstance(activity.object, dict) and isinstance(activity.target, str):
        object_id = activity.object.get("id")
        target_collection = activity.target

        # Check if the target collection exists and is owned by the actor
        collection = await db.collections.find_one(
            {"id": target_collection, "owner": activity.actor}
        )

        if collection:
            # Add the object to the collection
            await db.collections.update_one(
                {"id": target_collection}, {"$addToSet": {"items": object_id}}
            )

            # If the collection is an OrderedCollection, we might want to maintain order
            if collection.get("type") == "OrderedCollection":
                await db.collections.update_one(
                    {"id": target_collection}, {"$push": {"orderedItems": object_id}}
                )


async def handle_remove(activity: Activity):
    if isinstance(activity.object, dict) and isinstance(activity.target, str):
        object_id = activity.object.get("id")
        target_collection = activity.target

        # Check if the target collection exists and is owned by the actor
        collection = await db.collections.find_one(
            {"id": target_collection, "owner": activity.actor}
        )

        if collection:
            # Remove the object from the collection
            await db.collections.update_one(
                {"id": target_collection}, {"$pull": {"items": object_id}}
            )

            # If the collection is an OrderedCollection, remove from orderedItems as well
            if collection.get("type") == "OrderedCollection":
                await db.collections.update_one(
                    {"id": target_collection}, {"$pull": {"orderedItems": object_id}}
                )


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
            existing_object = await db.objects.find_one({"id": object_id})
            if existing_object:
                # Perform partial update
                for key, value in object_data.items():
                    if value is None:
                        existing_object.pop(key, None)
                    else:
                        existing_object[key] = value
                await db.objects.replace_one({"id": object_id}, existing_object)
    await forward_to_followers(activity)


async def handle_block(activity: Activity):
    blocked_actor = activity.object
    if isinstance(blocked_actor, str):
        await db.blocked.update_one(
            {"username": activity.actor},
            {"$addToSet": {"items": blocked_actor}},
            upsert=True,
        )


async def handle_delete(activity: Activity):
    object_id = activity.object
    if isinstance(object_id, str):
        existing_object = await db.objects.find_one({"id": object_id})
        if existing_object:
            tombstone = Tombstone(
                id=object_id,
                formerType=existing_object.get("type"),
                deleted=datetime.utcnow(),
            )
            await db.objects.replace_one({"id": object_id}, tombstone.dict())
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


async def handle_undo(activity: Activity):
    if isinstance(activity.object, dict):
        undone_activity_type = activity.object.get("type")
        if undone_activity_type == "Follow":
            await handle_unfollow(activity)
        elif undone_activity_type == "Like":
            await undo_like(activity)
        elif undone_activity_type == "Announce":
            await undo_announce(activity)


async def undo_like(activity: Activity):
    liked_object = activity.object.get("object")
    if isinstance(liked_object, str):
        await db.liked.update_one(
            {"username": activity.actor}, {"$pull": {"items": liked_object}}
        )


async def undo_announce(activity: Activity):
    announced_object = activity.object.get("object")
    if isinstance(announced_object, str):
        await db.announced.update_one(
            {"username": activity.actor}, {"$pull": {"items": announced_object}}
        )


async def handle_accept(activity: Activity):
    if isinstance(activity.object, dict) and activity.object.get("type") == "Follow":
        follower = activity.object.get("actor")
        followed = activity.actor
        await db.following.update_one(
            {"username": follower}, {"$addToSet": {"items": followed}}, upsert=True
        )


async def handle_reject(activity: Activity):
    if isinstance(activity.object, dict) and activity.object.get("type") == "Follow":
        follower = activity.object.get("actor")
        followed = activity.actor
        await db.following.update_one(
            {"username": follower}, {"$pull": {"items": followed}}
        )


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
    if verify_signature(
        public_key,
        sig_parts["signature"],
        request.method,
        request.url.path,
        request.headers,
    ):
        return
    else:
        raise HTTPException(
            status_code=401, detail=f"Invalid signature: {signature_header}"
        )


def create_signature(private_key, method, path, headers):
    signing_string = f"(request-target): {method.lower()} {path}\n"
    signing_string += "\n".join(f"{k.lower()}: {v}" for k, v in headers.items())

    signature = private_key.sign(
        signing_string.encode(), padding.PKCS1v15(), hashes.SHA256()
    )
    return base64.b64encode(signature).decode()


def verify_signature(public_key, signature, method, path, headers):
    signing_string = f"(request-target): {method.lower()} {path}\n"
    signing_string += "\n".join(f"{k.lower()}: {v}" for k, v in headers.items())

    try:
        public_key.verify(
            base64.b64decode(signature),
            signing_string.encode(),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


@app.post("/users", status_code=201)
async def create_user(username: str, email: EmailStr, password: str):
    hashed_password = get_password_hash(password)
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
    await db.users.insert_one(
        {
            "username": username,
            "email": email,
            "hashed_password": hashed_password,
        }
    )
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
        username = user["preferredUsername"]

        # De-duplicate activities
        existing_activity = await db.inboxes.find_one(
            {"username": username, "id": activity.id}
        )

        if not existing_activity:
            await db.inboxes.insert_one({**activity.dict(), "username": username})
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
