# 📦 DreamWeaver.AI Streamlit App Starter Kit

An AI-powered nutritional app designed to help you analyze and improve your health quality through advanced AI recommendations. 
https://vitatraxx.streamlit.app/

## Setup Instructions

### 1. Create and Activate a Virtual Environment

First, create a virtual environment:

```bash
python -m venv dw_venv
```

Activate the environment:

- **Windows:**

    ```bash
    .\dw_venv\Scripts\activate
    ```

- **macOS/Linux:**

    ```bash
    source dw_venv/bin/activate
    ```
### 2. Secrets Configuration

## To get a google Gemini API key
Go to https://aistudio.google.com/app/apikey, sign in with your google account and click 'Get API Key' -> 'Generate API Key'
To keep your API keys and other sensitive information secure, add them to the `secrets.toml` file.

## Example `secrets.toml` Configuration:**

```toml
[general]
GOOGLE_API_KEY = "your_api_key"
GEMINI_API_KEY = "your_api_key"
USDA_API_KEY = "your_api_key"
```
### 3. Install Requirements

Install the necessary dependencies:

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Application

To start the application, run the following command:

```bash
python -m streamlit run app.py
```

## Deployment

### Deploying on Streamlit Cloud

1. Visit the Streamlit sharing platform:

   [Streamlit Cloud](https://share.streamlit.io/)

2. Click the **Create App** button on the top right.

3. Choose **Yup, I have an app**.

![Deployment Step 1](docs/images/1.png)
*Figure 1: Selecting the repository and branch.*

4. Select the appropriate repository, branch, and starting script (`app.py`).

![Deployment Step 2](docs/images/2.png)
*Figure 2: Reviewing deployment settings.*

5. Your app will be deployed and accessible via a public URL.

### Add your api keys or env variables

6. On the list of apps level edit the app with hamburger menu.

![Deployment Step 3](docs/images/3.png)
*Figure 3: Reviewing deployment settings.*

7. Update the values accordingly with your `secrets.toml` file

![Deployment Step 4](docs/images/4.png)
*Figure 4: Reviewing deployment settings.*


## GitHub Codespaces

You can also launch this project in GitHub Codespaces for a seamless development experience:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

## Further Reading

Explore more about this project:

- [Resource 1](#)
- [Resource 2](#)
- [Resource 3](#)
