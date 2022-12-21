# Anti-Bait 📰

Uses Python ML to detect biased news.  
The data to train this program is provided in the file **news.csv**.  

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## Libraries used

- Numpy
- Pandas
- Sklearn
- Streamlit

## Local development (Windows)

```sh
py -m pip install --user virtualenv
```

```sh
py -m venv env
```

Authorize access

```sh
Set-ExecutionPolicy Unrestricted -Scope Process
```

```sh
.\env\Scripts\activate.ps1
```

## Installing libraries

```sh
py -m pip install [library-name]
```

## How to run unit tests

Run the command below.

```sh
py -m unittest -v fake-news.py
```

## How to use the UI

Run the command below.

```sh
py -m streamlit run gui.py

```

![image](https://user-images.githubusercontent.com/104475739/201845580-17f304f2-f776-4faf-a643-a11313d552dd.png)

The UI should be on <http://localhost:8501/>

## File information

- **fake-news.py**  
This file contains the model to detect biased news.  
When prompted, enter the headline in text form in the terminal to get either "UNBIASED" or "BIASED".

- **cm.py**  
This file contains the Confusion Matrix showing the validity of the model.
![image](https://user-images.githubusercontent.com/104475739/201575746-46eaeda6-5ce7-41ac-a9fe-0ced0acea80d.png)

- **gui.py**  
This file contains the code for GUI showing the model.
