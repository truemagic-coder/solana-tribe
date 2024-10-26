from pymongo import MongoClient
import pytest
from fastapi.testclient import TestClient
from dotenv import load_dotenv
import os

load_dotenv(".env.test")
from main import app, create_access_token # noqa: E402

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def clear_database():
    yield  # This line allows the tests to run
    # After all tests have run, clear the database
    client = MongoClient(os.getenv("MONGO_URL"))
    client.drop_database("solanatribe")


@pytest.fixture
def auth_headers(test_access_token):
    return {"Authorization": f"Bearer {test_access_token}"}


@pytest.fixture(scope="module")
def test_access_token():
    return create_access_token(data={"sub": "testuser"})


# User Management Tests
def test_create_user():
    response = client.post(
        "/users?username=newuser&email=new@example.com&password=newpassword"
    )
    assert response.status_code == 201
    assert response.json() == {"message": "User created successfully"}


def test_create_existing_user():
    client.post(
        "/users?username=existinguser&email=existing@example.com&password=password"
    )
    response = client.post(
        "/users?username=existinguser&email=existing@example.com&password=password"
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


# Authentication Tests
def test_login_for_access_token():
    client.post("/users?username=testuser&email=test@example.com&password=testpassword")
    response = client.post(
        "/token", data={"username": "testuser", "password": "testpassword"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


def test_login_invalid_credentials():
    response = client.post(
        "/token", data={"username": "testuser", "password": "wrongpassword"}
    )
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
    client.post(
        "/users?username=webfinger&email=webfinger@example.com&password=password"
    )
    response = client.get("/.well-known/webfinger?resource=acct:webfinger@example.com")
    assert response.status_code == 200
    assert "links" in response.json()


def test_webfinger_nonexistent_user():
    response = client.get(
        "/.well-known/webfinger?resource=acct:nonexistent@example.com"
    )
    assert response.status_code == 404


# Collections Tests
@pytest.mark.parametrize("collection", ["outbox", "followers", "following", "liked"])
def test_get_collection(collection):
    client.post(
        "/users?username=collectionuser&email=collection@example.com&password=password"
    )
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
        "object": {"type": "Note", "content": "This is a test note"},
    }
    response = client.post(
        "/actors/testuser/outbox", json=activity, headers=auth_headers
    )
    assert response.status_code == 201


def test_update_activity(auth_headers):
    # First, create an object
    create_activity = {
        "type": "Create",
        "actor": "https://example.com/actors/testuser",
        "object": {"type": "Note", "content": "This is a test note"},
    }
    create_response = client.post(
        "/actors/testuser/outbox", json=create_activity, headers=auth_headers
    )
    created_object_id = create_response.json()["id"]

    # Now, update the object
    update_activity = {
        "type": "Update",
        "actor": "https://example.com/actors/testuser",
        "object": {
            "id": created_object_id,
            "type": "Note",
            "content": "This is an updated test note",
        },
    }
    response = client.post(
        "/actors/testuser/outbox", json=update_activity, headers=auth_headers
    )
    assert response.status_code == 201


def test_delete_activity(auth_headers):
    # First, create an object
    create_activity = {
        "type": "Create",
        "actor": "https://example.com/actors/testuser",
        "object": {"type": "Note", "content": "This is a test note to be deleted"},
    }
    create_response = client.post(
        "/actors/testuser/outbox", json=create_activity, headers=auth_headers
    )
    created_object_id = create_response.json()["id"]

    # Now, delete the object
    delete_activity = {
        "type": "Delete",
        "actor": "https://example.com/actors/testuser",
        "object": created_object_id,
    }
    response = client.post(
        "/actors/testuser/outbox", json=delete_activity, headers=auth_headers
    )
    assert response.status_code == 201


def test_follow_activity(auth_headers):
    follow_activity = {
        "type": "Follow",
        "actor": "https://example.com/actors/testuser",
        "object": "https://example.com/actors/otheruser",
    }
    response = client.post(
        "/actors/testuser/outbox", json=follow_activity, headers=auth_headers
    )
    assert response.status_code == 201


def test_like_activity(auth_headers):
    like_activity = {
        "type": "Like",
        "actor": "https://example.com/actors/testuser",
        "object": "https://example.com/objects/somecontent",
    }
    response = client.post(
        "/actors/testuser/outbox", json=like_activity, headers=auth_headers
    )
    assert response.status_code == 201


def test_announce_activity(auth_headers):
    announce_activity = {
        "type": "Announce",
        "actor": "https://example.com/actors/testuser",
        "object": "https://example.com/objects/somecontent",
    }
    response = client.post(
        "/actors/testuser/outbox", json=announce_activity, headers=auth_headers
    )
    assert response.status_code == 201


# Server-to-Server Interaction Tests
def test_inbox_post(monkeypatch):
    # Mock the verify_http_signature function to always return True
    monkeypatch.setattr("main.verify_http_signature", lambda request: True)

    activity = {
        "type": "Create",
        "actor": "https://otherserver.com/actors/remoteuser",
        "object": {"type": "Note", "content": "This is a remote note"},
    }
    response = client.post("/actors/testuser/inbox", json=activity)
    assert response.status_code == 202


def test_shared_inbox_post(monkeypatch):
    # Mock the verify_http_signature function to always return True
    monkeypatch.setattr("main.verify_http_signature", lambda request: True)

    activity = {
        "type": "Announce",
        "actor": "https://otherserver.com/actors/remoteuser",
        "object": "https://example.com/objects/somecontent",
    }
    response = client.post("/shared_inbox", json=activity)
    assert response.status_code == 202


# Content Negotiation Tests
def test_content_negotiation_json_ld():
    headers = {"Accept": "application/ld+json"}
    response = client.get("/actors/testuser", headers=headers)
    assert response.status_code == 200
    assert (
        response.headers["Content-Type"]
        == 'application/ld+json; profile="https://www.w3.org/ns/activitystreams"'
    )


def test_content_negotiation_activity_json():
    headers = {"Accept": "application/activity+json"}
    response = client.get("/actors/testuser", headers=headers)
    assert response.status_code == 200
    assert (
        response.headers["Content-Type"]
        == 'application/ld+json; profile="https://www.w3.org/ns/activitystreams"'
    )


# Test context handling
def test_context_handling():
    response = client.get("/actors/testuser")
    assert response.status_code == 200
    assert "@context" in response.json()
    assert "https://www.w3.org/ns/activitystreams" in response.json()["@context"]


# Test activity validation
def test_activity_validation(monkeypatch, auth_headers):
    invalid_activity = {
        "type": "InvalidType",
        "actor": "https://example.com/actors/testuser",
        "object": "This is not a valid object",
    }
    response = client.post(
        "/actors/testuser/outbox", json=invalid_activity, headers=auth_headers
    )
    assert response.status_code == 422


# Test public key retrieval
def test_public_key_retrieval():
    response = client.get("/actors/testuser")
    assert response.status_code == 200
    assert "publicKey" in response.json()
    assert "id" in response.json()["publicKey"]
    assert "owner" in response.json()["publicKey"]
    assert "publicKeyPem" in response.json()["publicKey"]


# Test undo activity
def test_undo_activity(auth_headers):
    # First, create a Like activity
    like_activity = {
        "type": "Like",
        "actor": "https://example.com/actors/testuser",
        "object": "https://example.com/objects/somecontent",
    }
    like_response = client.post(
        "/actors/testuser/outbox", json=like_activity, headers=auth_headers
    )
    assert like_response.status_code == 201
    like_id = like_response.json()["id"]

    # Now, undo the Like activity
    undo_activity = {
        "type": "Undo",
        "actor": "https://example.com/actors/testuser",
        "object": like_id,
    }
    undo_response = client.post(
        "/actors/testuser/outbox", json=undo_activity, headers=auth_headers
    )
    assert undo_response.status_code == 201


if __name__ == "__main__":
    pytest.main()
