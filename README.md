# BACKGROUND_REMOVER

This repository contains  Flask API's based backend code that can be used to remove background from images.


## 👨🏻‍🔬 Set up and run The Server

### 📃 Clone

Clone the repository from GitHub.

```
$ git clone https://github.com/PriyanshuRj/marketplace.git
```



#### 📂 Create your .venv file

1. After cloning the project navigate to the project directory using `cd marketplace` command.
2. Generate `.venv` in the directory.
    - prerequisite `venv library installed`
    - use command `py -m venv .venv` in Windows and `python3 -m venv .venv` in linux.
3. To activate virtual environment just run `.venv\Scripts\activate` in windows and `source .venv\bin\activate` in linux.




### 💻 Install Dependencies and Run the Server

```
$ pip install -r requirements.txt
$ flask run \ py app.py (for windows) \ python3 app.py (for linux)
```
Now, use you can use the use the API's and try them out.


## ⚙️ Specification

### /imgs POST
This endpoint can be used to add a report for a given marketor mandi.

```http
POST https://localhost:5000/imgs HTTP/1.1
Content-type: multipart/form-data;

{
    "image" : <IMAGE>
}

Response:
{
    "DOWNLOADABLE_FILE" : <SAVE_RESPONSE_WITH_IMAGE_EXTENSIONS>
}

common image extensions - png, jpg, jpeg
```


### /zip POST
This endpoint can be used to add a report for a given marketor mandi.

```http
POST https://localhost:5000/zip HTTP/1.1
Content-type: multipart/form-data;

{
    "image" : <IMAGE_FILE>
}

Response:
{
    "DOWNLOADABLE_FILE" : <SAVE_RESPONSE_WITH_ZIP_EXTENSIONS>
}

common image extensions - zip, .7z, .rar, .tar.gz

## Screenshorts off API operations
![screen3](https://user-images.githubusercontent.com/76838551/186371923-e727bd15-44fe-4a7a-87a3-ae0119f4e1b9.jpg)
![Screenshot 2022-08-24 140524screen2](https://user-images.githubusercontent.com/76838551/186371937-c1421fab-4c4c-438a-aa26-88abfebaea72.jpg)
![screen1](https://user-images.githubusercontent.com/76838551/186371940-69d4685d-c6b0-4794-b25c-4037f6ed28f2.jpg)


