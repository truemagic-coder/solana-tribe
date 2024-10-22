# ActivityPub Server 

This implementation aims to be fully compliant with the ActivityPub server spec.

https://www.w3.org/TR/activitypub/

## Stack:
* FastAPI
* MongoDB
* Taskiq (Redis)

## Env Vars:
* MONGO_URL
* REDIS_URL
* SERVER_PRIVATE_KEY
* BASE_URL
* SECRET_KEY (for JWT)

This is a work in progress...

### TODO
* Implement SHDW drive storage and media uploading according to spec (/upload) and modify the activity base model
* Write full e2e test suite with inline spec comments
* Create CLI program with config file support
* Deploy to PyPi
