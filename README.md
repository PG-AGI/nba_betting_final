# NBA Betting Project Setup Guide

This guide outlines the steps to set up the NBA Betting project on your local machine.

## Step 1: Download and install Docker

Skipping this assuming you have Docker.

## Step 2: Build Docker Container and image

Run this command in your terminal to build Docker container

```bash
docker-compose up --build
```

Note:- it may take a while first time you run docker(approx 15 min).

## Step 4: Do some changes in the URL

When the docker is built this is the output you have to wait for:-
![Output](images/vs_link.png)

To open this link directly use shortcut `Ctrl + Click ` on the link then,

Change the url in your browser from `127.0.0.1:5000` to `127.0.0.1:80`

From This:-

![Original URL](images/Original_URL.png)

To This:-

![Change URL](images/URL_Change.png)
