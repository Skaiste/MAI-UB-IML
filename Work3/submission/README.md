### Team
- Skaiste Butkute
- Tatevik Davtyan
- Wiktoria Pejs

### Running script for the first time
These sections show how to create virtual environment for
our script and how to install dependencies
1. Open folder in terminal
```bash
cd <root_folder_of_project>/
```
2. Create virtual env
```bash
python3.9 -m venv venv/
```
3. Open virtual env
```bash
source venv/bin/activate
```
4. Install required dependencies
```bash
pip install -r requirements.txt
```
you can check if dependencies were installed by running next
command,it should print list with installed dependencies
```bash
pip list
```

### Run the script for a selected model
1. Open virtual env
```bash
source venv/bin/activate
```
2. Running the script.
When running the script you will have to input the path to the dataset file. If the relative path doesn't work, provide an absolute path.
```bash
python3.9 code/main.py
```
3. Close virtual env
```bash
deactivate
```