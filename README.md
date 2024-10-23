# Solana Tribe

This implementation aims to be a fully compliant ActivityPub server using JWTs and SHDW Drive.

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
* Write full e2e test suite with inline spec comments
* Update readme with checkmarks for all specs completed/supported
* Create CLI program with config file support
* Deploy to PyPi
