# Solana Tribe

This implementation aims to be a fully compliant ActivityPub server using SHDW Drive as media storage.

https://www.w3.org/TR/activitypub/

Current Status: Pre-Alpha (WIP)

## Stack:
* FastAPI
* MongoDB
* Taskiq (Redis)

## Env Vars:
* MONGO_URL
* REDIS_URL
* SERVER_PRIVATE_KEY
* BASE_URL
* HS256_SECRET_KEY or RS256_PRIVATE_KEY or RS256_PUBLIC_KEY

### TODO
* Implement SHDW drive storage and media uploading according to spec (/upload) and modify the activity base model
* Write full e2e test suite with inline spec comments
* Update readme with checkmarks for all specs completed/supported
* Create CLI program with config file support
* Deploy to PyPi
