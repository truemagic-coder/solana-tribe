import json
import pytest
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from rdflib import Graph
from rdflib.plugin import register, Parser, Serializer

load_dotenv(".env.test")

from main import app, db, create_access_token  # noqa: E402

# Register JSON-LD plugin
register("json-ld", Parser, "rdflib_jsonld.parser", "JsonLDParser")
register("json-ld", Serializer, "rdflib_jsonld.serializer", "JsonLDSerializer")

client = TestClient(app)

# Mock data
mock_data = {
    "users": {},
    "actors": {},
    "inboxes": {},
    "outboxes": {},
    "followers": {},
    "following": {},
    "liked": {},
    "objects": {},
    "shares": {},
}

# Mock database operations
@pytest.fixture(autouse=True)
def mock_db(monkeypatch):
    async def mock_find_one(collection, query):
        return mock_data[collection].get(query.get("username") or query.get("preferredUsername"))

    async def mock_insert_one(collection, document):
        mock_data[collection][document["username"]] = document

    async def mock_update_one(collection, query, update, upsert=False):
        item = mock_data[collection].get(query.get("username"))
        if item:
            item.update(update["$set"])
        elif upsert:
            mock_data[collection][query.get("username")] = update["$set"]

    async def mock_count_documents(*args, **kwargs):
        return len(mock_data[args[0]])

    monkeypatch.setattr(db.users, "find_one", mock_find_one)
    monkeypatch.setattr(db.actors, "find_one", mock_find_one)
    monkeypatch.setattr(db.users, "insert_one", mock_insert_one)
    monkeypatch.setattr(db.actors, "insert_one", mock_insert_one)
    monkeypatch.setattr(db.followers, "update_one", mock_update_one)
    monkeypatch.setattr(db.following, "update_one", mock_update_one)
    monkeypatch.setattr(db.inboxes, "count_documents", mock_count_documents)
    monkeypatch.setattr(db.outboxes, "count_documents", mock_count_documents)
    monkeypatch.setattr(db.followers, "count_documents", mock_count_documents)
    monkeypatch.setattr(db.following, "count_documents", mock_count_documents)
    monkeypatch.setattr(db.liked, "count_documents", mock_count_documents)
    monkeypatch.setattr(db.shares, "count_documents", mock_count_documents)

@pytest.fixture
def auth_headers():
    access_token = create_access_token(data={"sub": "testuser"})
    return {"Authorization": f"Bearer {access_token}"}

# User Management Tests
def test_create_user():
    response = client.post("/users?username=newuser&email=new@example.com&password=newpassword")
    assert response.status_code == 201
    assert response.json() == {"message": "User created successfully"}

def test_create_existing_user():
    client.post("/users?username=existinguser&email=existing@example.com&password=password")
    response = client.post("/users?username=existinguser&email=existing@example.com&password=password")
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]

# Authentication Tests
def test_login_for_access_token():
    client.post("/users?username=testuser&email=test@example.com&password=testpassword")
    response = client.post("/token", data={"username": "testuser", "password": "testpassword"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_login_invalid_credentials():
    response = client.post("/token", data={"username": "testuser", "password": "wrongpassword"})
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]

def test_protected_route(auth_headers):
    response = client.get("/actors/testuser/inbox", headers=auth_headers)
    assert response.status_code == 200

def test_protected_route_no_token():
    response = client.get("/actors/testuser/inbox")
    assert response.status_code == 401

# Actor Endpoints Tests
def test_get_actor():
    client.post("/users?username=testactor&email=actor@example.com&password=password")
    response = client.get("/actors/testactor")
    assert response.status_code == 200
    assert response.json()["preferredUsername"] == "testactor"

def test_get_nonexistent_actor():
    response = client.get("/actors/nonexistent")
    assert response.status_code == 404

def test_webfinger():
    client.post("/users?username=webfinger&email=webfinger@example.com&password=password")
    response = client.get("/.well-known/webfinger?resource=acct:webfinger@example.com")
    assert response.status_code == 200
    assert "links" in response.json()

def test_webfinger_nonexistent_user():
    response = client.get("/.well-known/webfinger?resource=acct:nonexistent@example.com")
    assert response.status_code == 404

# Collections Tests
@pytest.mark.parametrize("collection", ["outbox", "followers", "following", "liked"])
def test_get_collection(collection):
    client.post("/users?username=collectionuser&email=collection@example.com&password=password")
    response = client.get(f"/actors/collectionuser/{collection}")
    assert response.status_code == 200
    assert response.json()["type"] == "OrderedCollection"

@pytest.mark.parametrize("collection", ["outbox", "followers", "following", "liked"])
def test_get_collection_page(collection):
    client.post("/users?username=pageuser&email=page@example.com&password=password")
    response = client.get(f"/actors/pageuser/{collection}?page=1")
    assert response.status_code == 200
    assert response.json()["type"] == "OrderedCollectionPage"

def test_get_shares():
    response = client.get("/objects/testobject/shares")
    assert response.status_code == 200
    assert response.json()["type"] == "OrderedCollection"

# Activity Tests
def test_create_activity(auth_headers):
    activity = {
        "type": "Create",
        "actor": "https://example.com/actors/testuser",
        "object": {
            "type": "Note",
            "content": "This is a test note"
        }
    }
    response = client.post("/actors/testuser/outbox", json=activity, headers=auth_headers)
    assert response.status_code == 201

def test_update_activity(auth_headers):
    # First, create an object
    create_activity = {
        "type": "Create",
        "actor": "https://example.com/actors/testuser",
        "object": {
            "type": "Note",
            "content": "This is a test note"
        }
    }
    create_response = client.post("/actors/testuser/outbox", json=create_activity, headers=auth_headers)
    created_object_id = create_response.json()["id"]

    # Now, update the object
    update_activity = {
        "type": "Update",
        "actor": "https://example.com/actors/testuser",
        "object": {
            "id": created_object_id,
            "type": "Note",
            "content": "This is an updated test note",
        }
    }
    response = client.post("/actors/testuser/outbox", json=update_activity, headers=auth_headers)
    assert response.status_code == 201

def test_delete_activity(auth_headers):
    # First, create an object
    create_activity = {
        "type": "Create",
        "actor": "https://example.com/actors/testuser",
        "object": {
            "type": "Note",
            "content": "This is a test note to be deleted"
        }
    }
    create_response = client.post("/actors/testuser/outbox", json=create_activity, headers=auth_headers)
    created_object_id = create_response.json()["id"]

    # Now, delete the object
    delete_activity = {
        "type": "Delete",
        "actor": "https://example.com/actors/testuser",
        "object": created_object_id
    }
    response = client.post("/actors/testuser/outbox", json=delete_activity, headers=auth_headers)
    assert response.status_code == 201

def test_follow_activity(auth_headers):
    follow_activity = {
        "type": "Follow",
        "actor": "https://example.com/actors/testuser",
        "object": "https://example.com/actors/otheruser"
    }
    response = client.post("/actors/testuser/outbox", json=follow_activity, headers=auth_headers)
    assert response.status_code == 201

def test_like_activity(auth_headers):
    like_activity = {
        "type": "Like",
        "actor": "https://example.com/actors/testuser",
        "object": "https://example.com/objects/somecontent"
    }
    response = client.post("/actors/testuser/outbox", json=like_activity, headers=auth_headers)
    assert response.status_code == 201

def test_announce_activity(auth_headers):
    announce_activity = {
        "type": "Announce",
        "actor": "https://example.com/actors/testuser",
        "object": "https://example.com/objects/somecontent"
    }
    response = client.post("/actors/testuser/outbox", json=announce_activity, headers=auth_headers)
    assert response.status_code == 201

# Server-to-Server Interaction Tests
def test_inbox_post(monkeypatch):
    # Mock the verify_http_signature function to always return True
    monkeypatch.setattr("main.verify_http_signature", lambda request: True)
    
    activity = {
        "type": "Create",
        "actor": "https://otherserver.com/actors/remoteuser",
        "object": {
            "type": "Note",
            "content": "This is a remote note"
        }
    }
    response = client.post("/actors/testuser/inbox", json=activity)
    assert response.status_code == 202

def test_shared_inbox_post(monkeypatch):
    # Mock the verify_http_signature function to always return True
    monkeypatch.setattr("main.verify_http_signature", lambda request: True)
    
    activity = {
        "type": "Announce",
        "actor": "https://otherserver.com/actors/remoteuser",
        "object": "https://example.com/objects/somecontent"
    }
    response = client.post("/shared_inbox", json=activity)
    assert response.status_code == 202

# Content Negotiation Tests
def test_content_negotiation_json_ld():
    headers = {"Accept": "application/ld+json"}
    response = client.get("/actors/testuser", headers=headers)
    assert response.status_code == 200
    assert response.headers["Content-Type"] == 'application/ld+json; profile="https://www.w3.org/ns/activitystreams"'

def test_content_negotiation_activity_json():
    headers = {"Accept": "application/activity+json"}
    response = client.get("/actors/testuser", headers=headers)
    assert response.status_code == 200
    assert response.headers["Content-Type"] == 'application/ld+json; profile="https://www.w3.org/ns/activitystreams"'

# HTTP Signatures Tests
def test_create_signed_headers():
    from main import create_signed_headers
    headers = create_signed_headers("POST", "https://example.com/inbox", {"type": "Create"})
    assert "Signature" in headers
    assert "Date" in headers
    assert "Digest" in headers

def test_verify_http_signature(monkeypatch):
    from main import verify_http_signature
    
    # Mock the necessary components for the test
    class MockRequest:
        headers = {
            "Signature": "keyId=\"https://example.com/actors/testuser#main-key\",algorithm=\"rsa-sha256\",headers=\"(request-target) host date digest\",signature=\"mock_signature\""
        }
        method = "POST"
        url = type("URL", (), {"path": "/inbox"})()

    monkeypatch.setattr("httpx.AsyncClient.get", lambda *args, **kwargs: type("Response", (), {"json": lambda: {"publicKey": {"publicKeyPem": "mock_public_key"}}})())
    monkeypatch.setattr("main.verify_headers", lambda *args, **kwargs: True)

    result = verify_http_signature(MockRequest())
    assert result is None  # If no exception is raised, the verification is successful

# Error Handling Tests
def test_method_not_allowed():
    response = client.put("/actors/testuser")  # PUT is not allowed
    assert response.status_code == 405
    assert response.json()["detail"] == "Method Not Allowed"

def test_not_found():
    response = client.get("/nonexistent/endpoint")
    assert response.status_code == 404
    assert response.json()["detail"] == "Not Found"

# Pagination Tests
def test_collection_pagination():
    # Add multiple items to a collection
    for i in range(25):  # Assuming 20 items per page
        mock_data["outboxes"]["testuser"] = mock_data["outboxes"].get("testuser", []) + [{"id": f"item{i}"}]

    response = client.get("/actors/testuser/outbox")
    assert response.status_code == 200
    assert response.json()["type"] == "OrderedCollection"
    assert "first" in response.json()
    assert "last" in response.json()

    # Check first page
    first_page_response = client.get(response.json()["first"])
    assert first_page_response.status_code == 200
    assert first_page_response.json()["type"] == "OrderedCollectionPage"
    assert len(first_page_response.json()["orderedItems"]) == 20
    assert "next" in first_page_response.json()

    # Check second page
    second_page_response = client.get(first_page_response.json()["next"])
    assert second_page_response.status_code == 200
    assert second_page_response.json()["type"] == "OrderedCollectionPage"
    assert len(second_page_response.json()["orderedItems"]) == 5
    assert "prev" in second_page_response.json()

# Test rate limiting (if implemented)
def test_rate_limiting():
    for _ in range(100):  # Assuming rate limit is less than 100 requests per minute
        response = client.get("/actors/testuser")
    assert response.status_code in [200, 429]  # Either successful or too many requests

# Test media upload (if implemented)
def test_media_upload(auth_headers):
    with open("test_image.jpg", "rb") as image_file:
        files = {"file": ("test_image.jpg", image_file, "image/jpeg")}
        response = client.post("/media", files=files, headers=auth_headers)
    assert response.status_code == 201
    assert "url" in response.json()

# Test federation (if implemented)
def test_federation(monkeypatch):
    # Mock the send_to_remote_inbox function
    monkeypatch.setattr("main.send_to_remote_inbox", lambda activity, inbox_url: True)
    
    activity = {
        "type": "Create",
        "actor": "https://example.com/actors/testuser",
        "object": {
            "type": "Note",
            "content": "This is a federated note"
        },
        "to": ["https://remoteserver.com/actors/remoteuser"]
    }
    response = client.post("/actors/testuser/outbox", json=activity, headers=auth_headers)
    assert response.status_code == 201

# Test object retrieval
def test_get_object():
    # First, create an object
    create_activity = {
        "type": "Create",
        "actor": "https://example.com/actors/testuser",
        "object": {
            "type": "Note",
            "content": "This is a test note for retrieval"
        }
    }
    create_response = client.post("/actors/testuser/outbox", json=create_activity, headers=auth_headers)
    created_object_id = create_response.json()["object"]["id"]

    # Now, retrieve the object
    response = client.get(f"/objects/{created_object_id.split('/')[-1]}")
    assert response.status_code == 200
    assert response.json()["type"] == "Note"
    assert response.json()["content"] == "This is a test note for retrieval"

# Test context handling
def test_context_handling():
    response = client.get("/actors/testuser")
    assert response.status_code == 200
    assert "@context" in response.json()
    assert "https://www.w3.org/ns/activitystreams" in response.json()["@context"]

# Test security headers
def test_security_headers():
    response = client.get("/actors/testuser")
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

# Test JSON-LD compaction (if implemented)
def test_jsonld_compaction():
    response = client.get("/actors/testuser")
    assert response.status_code == 200

    # Create an RDF graph from the response JSON
    g = Graph().parse(data=response.text, format="json-ld")

    # Compact the graph using the ActivityStreams context
    context = "https://www.w3.org/ns/activitystreams"
    compacted = g.serialize(format="json-ld", context=context, compact=True)

    # Parse the compacted JSON-LD
    compacted_json = json.loads(compacted)

    # Assertions
    assert "@context" in compacted_json
    assert compacted_json["@context"] == context
    assert "type" in compacted_json
    assert "id" in compacted_json
    assert "preferredUsername" in compacted_json

# Test activity delivery to followers
def test_activity_delivery_to_followers(monkeypatch):
    # Mock the send_to_remote_inbox function
    monkeypatch.setattr("main.send_to_remote_inbox", lambda activity, inbox_url: True)
    
    # Add a follower
    mock_data["followers"]["testuser"] = ["https://remoteserver.com/actors/follower"]
    
    activity = {
        "type": "Create",
        "actor": "https://example.com/actors/testuser",
        "object": {
            "type": "Note",
            "content": "This is a note for followers"
        }
    }
    response = client.post("/actors/testuser/outbox", json=activity, headers=auth_headers)
    assert response.status_code == 201

# Test activity validation
def test_activity_validation():
    invalid_activity = {
        "type": "InvalidType",
        "actor": "https://example.com/actors/testuser",
        "object": "This is not a valid object"
    }
    response = client.post("/actors/testuser/outbox", json=invalid_activity, headers=auth_headers)
    assert response.status_code == 400

# Test public key retrieval
def test_public_key_retrieval():
    response = client.get("/actors/testuser")
    assert response.status_code == 200
    assert "publicKey" in response.json()
    assert "id" in response.json()["publicKey"]
    assert "owner" in response.json()["publicKey"]
    assert "publicKeyPem" in response.json()["publicKey"]

# Test actor update
def test_actor_update(auth_headers):
    update_data = {
        "name": "Updated Name",
        "summary": "This is an updated summary"
    }
    response = client.patch("/actors/testuser", json=update_data, headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["name"] == "Updated Name"
    assert response.json()["summary"] == "This is an updated summary"

# Test actor deletion
def test_actor_deletion(auth_headers):
    response = client.delete("/actors/testuser", headers=auth_headers)
    assert response.status_code == 204

    # Verify that the actor is no longer retrievable
    get_response = client.get("/actors/testuser")
    assert get_response.status_code == 404

# Test undo activity
def test_undo_activity(auth_headers):
    # First, create a Like activity
    like_activity = {
        "type": "Like",
        "actor": "https://example.com/actors/testuser",
        "object": "https://example.com/objects/somecontent"
    }
    like_response = client.post("/actors/testuser/outbox", json=like_activity, headers=auth_headers)
    assert like_response.status_code == 201
    like_id = like_response.json()["id"]

    # Now, undo the Like activity
    undo_activity = {
        "type": "Undo",
        "actor": "https://example.com/actors/testuser",
        "object": like_id
    }
    undo_response = client.post("/actors/testuser/outbox", json=undo_activity, headers=auth_headers)
    assert undo_response.status_code == 201

# Test blocked actors (if implemented)
def test_blocked_actors(auth_headers, monkeypatch):
    # Mock the is_blocked function
    monkeypatch.setattr("main.is_blocked", lambda actor: actor == "https://example.com/actors/blockeduser")

    blocked_activity = {
        "type": "Create",
        "actor": "https://example.com/actors/blockeduser",
        "object": {
            "type": "Note",
            "content": "This should be blocked"
        }
    }
    response = client.post("/actors/testuser/inbox", json=blocked_activity)
    assert response.status_code == 403

# Test rate limiting for specific endpoints (if implemented)
def test_rate_limiting_specific_endpoints():
    for _ in range(50):  # Assuming a lower rate limit for the inbox
        response = client.post("/actors/testuser/inbox", json={"type": "Create"})
    assert response.status_code in [202, 429]  # Either accepted or too many requests

# Add more tests as needed for your specific implementation

if __name__ == "__main__":
    pytest.main()
