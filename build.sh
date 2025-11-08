#!/bin/sh
(cd frontend/ && npm install && npm run build)
(cd backend/ && pip3 install -r requirements.txt)
