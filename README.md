# ActivityPub Server 

This implementation aims to be fully compliant with the ActivityPub server spec.

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
* Write full test suite
* Create CLI program with config file support
* Deploy to PyPi
