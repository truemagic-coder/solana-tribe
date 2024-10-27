# Solana Tribe

This implementation aims to be a fully compliant ActivityPub server using JWTs with SHDW Drive and IPFS support.

https://www.w3.org/TR/activitypub/

Current Status: Pre-Alpha (WIP)

## Stack:
* FastAPI
* MongoDB

## Env Vars:
* MONGO_URL
* REDIS_URL
* SERVER_PRIVATE_KEY
* BASE_URL
* HS256_SECRET_KEY
* RS256_PRIVATE_KEY
* RS256_PUBLIC_KEY
* JWT_ALGORITHM (optional - defaults to HS256)

### TODO
* Implement SHDW drive storage and media uploading according to spec (/upload) and modify the activity base model with tests

## Implementation Status

* [ ] Shared server- & client-to-server
  * [x] Inbox GET
  * [x] Outbox GET
  * [x] Shared inbox GET
  * [x] Resource GET
    * [x] Object
    * [x] Actor
    * [x] Activity
    * [x] Collections
      * [x] Special collections
        * [x] Inbox
        * [x] Outbox
        * [x] Followers
        * [x] Following
        * [x] Liked
        * [x] Shares
      * [x] Misc collections (of activities)
      * [x] Pagination
    * [ ] Relay requests for remote objects
    * [ ] Response code 410 for Tombstones
  * [x] Security
    * [x] Permission-based filtering
* [ ] Server-to-server
  * [x] Inbox POST
    * [x] Activity side-effects
      * [x] Create
      * [x] Update
      * [x] Delete
      * [x] Follow
      * [x] Accept
      * [x] Reject
      * [x] Add
      * [x] Remove
      * [x] Like
      * [x] Announce
      * [x] Undo
      * [x] Block
    * [ ] Security
      * [x] Signature validation
      * [ ] Honor recipient blocklist
    * [ ] Recursive resolution of related objects
    * [ ] Forwarding from inbox
  * [ ] Shared inbox POST
    * [ ] Delivery to targeted local inboxes
  * [x] Delivery
    * [x] Request signing
    * [x] Addressing
      * [x] Shared inbox optmization
      * [ ] Direct delivery to local inboxes
    * [x] Redelivery attempts
* [ ] Client-to-server
  * [x] Outbox POST
    * [ ] Auto-Create for bare objects
    * [x] Activity side-effects
      * [x] Create
      * [x] Update
      * [x] Delete
      * [x] Follow
      * [x] Accept
      * [x] Reject
      * [x] Add
      * [x] Remove
      * [x] Like
      * [x] Announce
      * [x] Undo
      * [x] Block
  * [ ] Media upload
* [ ] Other
  * [x] Actor creation
    * [x] Key generation
  * [x] Security
    * [x] Verification
    * [ ] localhost block
    * [ ] Recursive object resolution depth limit
  * [ ] Related standards
    * [x] http-signature
    * [x] webfinger
    * [x] json-ld
      * [ ] Context cache
    * [ ] nodeinfo
    * [ ] Linked data signatures
  * [x] Storage model (denormalized MongoDB)
    * [ ] Index coverage for all queries
    * [ ] Fully interchangeable with documented API