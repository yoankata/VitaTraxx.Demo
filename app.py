import streamlit as st
import speech_recognition as sr
import requests
import sqlite3
import google.generativeai as genai
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gtts import gTTS
import io
import base64
import time
import random
from usda.client import UsdaClient
import numpy as np

# Configure your API keys
USDA_API_KEY = st.secrets.general.USDA_API_KEY
GEMINI_API_KEY = st.secrets.general.GEMINI_API_KEY

# Initialize USDA client
usda_client = UsdaClient(USDA_API_KEY)

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize SQLite database and close connection
#conn = sqlite3.connect('nutrition_data.db')#conn.close()
with sqlite3.connect('nutrition_data.db') as conn:  # Connection automatically closed
    c = conn.cursor()
    # Create table if it doesn't exist (expanded to include more nutrients)
    c.execute('''CREATE TABLE IF NOT EXISTS food_log
                (date TEXT, food TEXT, calories REAL, protein REAL, carbs REAL, fat REAL,
                vitamin_a REAL, vitamin_c REAL, vitamin_d REAL, vitamin_e REAL, vitamin_k REAL,
                thiamin REAL, riboflavin REAL, niacin REAL, vitamin_b6 REAL, folate REAL, vitamin_b12 REAL,
                calcium REAL, iron REAL, magnesium REAL, phosphorus REAL, potassium REAL, sodium REAL, zinc REAL,
                copper REAL, manganese REAL, selenium REAL, omega3 REAL,
                gluten REAL, mercury REAL, oxalates REAL, lectins REAL)''')

# RDA values (customize based on user profile)
rda = {
    'calories': 2000, 'protein': 50, 'carbs': 300, 'fat': 65,
    'vitamin_a': 900, 'vitamin_c': 90, 'vitamin_d': 20, 'vitamin_e': 15,
    'vitamin_k': 120, 'thiamin': 1.2, 'riboflavin': 1.3, 'niacin': 16,
    'vitamin_b6': 1.7, 'folate': 400, 'vitamin_b12': 2.4, 'calcium': 1000,
    'iron': 18, 'magnesium': 400, 'phosphorus': 700, 'potassium': 3400,
    'sodium': 2300, 'zinc': 11, 'copper': 0.9, 'manganese': 2.3,
    'selenium': 55, 'omega3': 1.6
}

# Define the nutrients
nutrients = {
    'calories': 'Energy', 'protein': 'Protein', 'carbs': 'Carbohydrate, by difference',
    'fat': 'Total lipid (fat)', 'vitamin_a': 'Vitamin A, RAE', 'vitamin_c': 'Vitamin C, total ascorbic acid',
    'vitamin_d': 'Vitamin D (D2 + D3)', 'vitamin_e': 'Vitamin E (alpha-tocopherol)',
    'vitamin_k': 'Vitamin K (phylloquinone)', 'thiamin': 'Thiamin', 'riboflavin': 'Riboflavin',
    'niacin': 'Niacin', 'vitamin_b6': 'Vitamin B-6', 'folate': 'Folate, total',
    'vitamin_b12': 'Vitamin B-12', 'calcium': 'Calcium, Ca', 'iron': 'Iron, Fe',
    'magnesium': 'Magnesium, Mg', 'phosphorus': 'Phosphorus, P', 'potassium': 'Potassium, K',
    'sodium': 'Sodium, Na', 'zinc': 'Zinc, Zn', 'copper': 'Copper, Cu', 'manganese': 'Manganese, Mn',
    'selenium': 'Selenium, Se', 'omega3': 'Fatty acids, total polyunsaturated',
    'gluten': 'Gluten', 'mercury': 'Mercury', 'oxalates': 'Oxalic acid', 'lectins': 'Lectins'
}
    
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)

        try:
            # Capture the audio from the microphone
            audio = r.listen(source, timeout=5)
            # Convert speech to text
            text = r.recognize_google(audio)
            st.success(f"You said: {text}")
            # CSS for animations
            #st.markdown("🎙️ Listening...", unsafe_allow_html=True)
            st.markdown("""
            <style>
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.2); }
                    100% { transform: scale(1); }
                }
                .animated-mic {
                    display: inline-block;
                    animation: pulse 1s infinite;
                }
            </style>
            """, unsafe_allow_html=True)
            st.markdown('<div class="animated-mic">🎙️</div> Listening...', unsafe_allow_html=True)
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError:
            st.error("Could not request results from the service. Check your internet connection.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            st.markdown("", unsafe_allow_html=True)

    return text


def get_nutrition_info(food):
    # url = f"https://api.nal.usda.gov/fdc/v1/foods/search?api_key={USDA_API_KEY}&query={food}"
    # response = requests.get(url)
    # if response.status_code == 200:
    #     data = response.json()
    #     if data['foods']:
    #         return data['foods'][0]['foodNutrients']
    # return None

    # Search for the food item using the UsdaClient
    response = usda_client.search_foods(food, max=5)
    # Check if there are any results
    if response['foods']:
        # Extract the food nutrients from the first result
        return response['foods'][0]['foodNutrients']
    return None

def log_food(food, nutrition_info):
    date = datetime.now().strftime("%Y-%m-%d")
    nutrients = {
        'calories': 'Energy', 'protein': 'Protein', 'carbs': 'Carbohydrate, by difference',
        'fat': 'Total lipid (fat)', 'vitamin_a': 'Vitamin A, RAE', 'vitamin_c': 'Vitamin C, total ascorbic acid',
        'vitamin_d': 'Vitamin D (D2 + D3)', 'vitamin_e': 'Vitamin E (alpha-tocopherol)',
        'vitamin_k': 'Vitamin K (phylloquinone)', 'thiamin': 'Thiamin', 'riboflavin': 'Riboflavin',
        'niacin': 'Niacin', 'vitamin_b6': 'Vitamin B-6', 'folate': 'Folate, total',
        'vitamin_b12': 'Vitamin B-12', 'calcium': 'Calcium, Ca', 'iron': 'Iron, Fe',
        'magnesium': 'Magnesium, Mg', 'phosphorus': 'Phosphorus, P', 'potassium': 'Potassium, K',
        'sodium': 'Sodium, Na', 'zinc': 'Zinc, Zn', 'copper': 'Copper, Cu', 'manganese': 'Manganese, Mn',
        'selenium': 'Selenium, Se', 'omega3': 'Fatty acids, total polyunsaturated',
        'gluten': 'Gluten', 'mercury': 'Mercury', 'oxalates': 'Oxalic acid', 'lectins': 'Lectins'
    }
    
    
    values = [date, food] + [next((n['value'] for n in nutrition_info if n['nutrientName'] == nutrient), 0) for nutrient in nutrients.values()]
    
    placeholders = ', '.join('?' * len(values))
    
    with sqlite3.connect('nutrition_data.db') as conn:  # Connection automatically closed
        c = conn.cursor()
        c.execute(f"INSERT INTO food_log VALUES ({placeholders})", values)
    conn.commit()

def get_daily_totals():
    date = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect('nutrition_data.db') as conn:  # Connection automatically closed
        c = conn.cursor()
        c.execute("SELECT * FROM food_log WHERE date = ?", (date,))
        columns = [description[0] for description in c.description]
        data = c.fetchall()
        df = pd.DataFrame(data, columns=columns)
        return df.sum().to_dict()
    return None


def get_food_items_from_db():
    with sqlite3.connect('nutrition_data.db') as conn:  # Connection automatically closed
        c = conn.cursor()
        c.execute("SELECT DISTINCT food FROM food_log")
        food_items = [row[0] for row in c.fetchall()]
        return food_items
    return None


def get_nutrient_data(food_item):
    with sqlite3.connect('nutrition_data.db') as conn:  # Connection automatically closed
        c = conn.cursor()
        c.execute("SELECT * FROM food_log WHERE food = ?", (food_item,))
        data = c.fetchone()
        if data:
            return {nutrient: data[i+1] for i, nutrient in enumerate(nutrients.keys())}
    return None


def create_nutrient_drill_down(data, selected_food):
    fig = go.Figure(data=[
        go.Bar(name=nutrient_name, x=[selected_food], y=[data[nutrient]])  # Extract the value from the data
        for nutrient, nutrient_name in nutrients.items() if data.get(nutrient) is not None
    ])
    
    fig.update_layout(
        title=f"Nutrient Contribution for {selected_food}",
        xaxis_title="Food Item",
        yaxis_title="Nutrient Content",
        barmode='group'
    )
    
    return fig

def get_cumulative_totals(days=7):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days-1)
    with sqlite3.connect('nutrition_data.db') as conn:  # Connection automatically closed
        c = conn.cursor()
        c.execute("SELECT * FROM food_log WHERE date BETWEEN ? AND ?", (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
        columns = [description[0] for description in c.description]
        data = c.fetchall()
        df = pd.DataFrame(data, columns=columns)
        return df.sum().to_dict()
    return None

def calculate_nutrient_scores(totals):
    nutrient_score = 0
    antinutrient_score = 0
    nutrient_count = 0
    antinutrient_count = 0

    for nutrient, value in totals.items():
        if nutrient in rda:
            progress = min(value / rda[nutrient], 1)
            nutrient_score += progress
            nutrient_count += 1
        elif nutrient in ['gluten', 'mercury', 'oxalates', 'lectins']:
            # For antinutrients, lower is better
            antinutrient_score += 1 - min(value / 100, 1)  # Assuming a threshold of 100 for antinutrients
            antinutrient_count += 1

    avg_nutrient_score = nutrient_score / nutrient_count if nutrient_count > 0 else 0
    avg_antinutrient_score = antinutrient_score / antinutrient_count if antinutrient_count > 0 else 1

    total_score = (avg_nutrient_score + avg_antinutrient_score) / 2

    return {
        'nutrient_score': avg_nutrient_score * 100,
        'antinutrient_score': avg_antinutrient_score * 100,
        'total_score': total_score * 100
    }

def get_missing_nutrients(daily_totals):
    missing = {}
    for nutrient, value in daily_totals.items():
        if nutrient in rda:
            if value < rda[nutrient]:
                missing[nutrient] = rda[nutrient] - value
    return missing

def get_recommended_foods(missing_nutrients):
    # recommended_foods = []
    # for nutrient, amount in missing_nutrients.items():
    #     search_results = usda_client.list_foods(5, sort=nutrient, sort_order='desc')
    #     for food in search_results:
    #         recommended_foods.append(food.description)
#def get_optimum_foods(missing_nutrients):
    recommended_foods = []
    for nutrient, amount in missing_nutrients.items():
        url = f"https://api.nal.usda.gov/fdc/v1/foods/search?api_key={USDA_API_KEY}&sortField={nutrient}&sortDirection=desc&pageSize=5"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for food in data['foods']:
                recommended_foods.append(food['description'])
    return list(set(recommended_foods))  # Remove duplicates    



def generate_instacart(recommended_foods):
    # Mock function to generate InstaCart
    stores = ["Walmart", "Kroger", "Whole Foods", "Costco", "Target"]
    return {store: random.sample(recommended_foods, min(3, len(recommended_foods))) for store in stores}

def generate_recipe(ingredients):
    prompt = f"""Create a healthy recipe using some or all of these ingredients: {', '.join(ingredients)}. 
                Provide a title, list of ingredients, 
                and step-by-step instructions. 
                Keep your answer short and bulleted."""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

def get_nutrition_advice(daily_totals, weekly_totals, scores):
    prompt = f"""As a nutrition coach, provide advice based on these daily totals: {daily_totals}. 
                Weekly averages: {weekly_totals}. The user's nutrient score is {scores['nutrient_score']:.2f}%, 
                antinutrient score is {scores['antinutrient_score']:.2f}%, 
                and total food score is {scores['total_score']:.2f}%.
                The nutritional advice should consist of reporting the totals and scores and what type of nutrients
                are required to achieve optimal RDA. Keep it short and bulleted."""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    return fp

def autoplay_audio(file):
    audio_base64 = base64.b64encode(file.getvalue()).decode()
    audio_tag = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    st.markdown(audio_tag, unsafe_allow_html=True)

def create_progress_bars(daily_totals):
    for nutrient, value in daily_totals.items():
        if nutrient in rda:
            progress = min(value / rda[nutrient], 1)
            st.progress(progress, text=f"{nutrient.capitalize()}: {value:.2f}/{rda[nutrient]} {get_unit(nutrient)}")

def get_unit(nutrient):
    if nutrient in ['calories']:
        return 'kcal'
    elif nutrient in ['protein', 'carbs', 'fat']:
        return 'g'
    elif nutrient in ['vitamin_a', 'vitamin_c', 'vitamin_e', 'vitamin_k', 'thiamin', 'riboflavin', 'niacin', 'vitamin_b6', 'folate', 'vitamin_b12']:
        return 'mg'
    elif nutrient in ['vitamin_d']:
        return 'µg'
    elif nutrient in ['calcium', 'iron', 'magnesium', 'phosphorus', 'potassium', 'sodium', 'zinc']:
        return 'mg'
    elif nutrient in ['copper', 'manganese', 'selenium']:
        return 'µg'
    else:
        return ''

def create_nutrient_plot(daily_totals):
    df = pd.DataFrame(list(daily_totals.items()), columns=['Nutrient', 'Value'])
    fig = px.bar(df, x='Nutrient', y='Value', title='Daily Nutrient Intake')
    st.plotly_chart(fig)

def generate_recipe(ingredients):
    prompt = f"""Create a healthy recipe using some or all of these ingredients: {', '.join(ingredients)}. 
                Provide a title, list of ingredients, and step-by-step instructions."""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

def mock_delivery_update():
    statuses = [
        "Order received",
        "Preparing your items",
        "Driver en route to store",
        "Shopping in progress",
        "Items packed and ready",
        "Driver on the way",
        "Arriving soon",
        "Delivered"
    ]
    return random.choice(statuses)

# # Simulating data from a continuous monitor
# def get_monitor_data(days=1):
#     dates = pd.date_range(end=pd.Timestamp.now(), periods=days*288, freq='5T')
#     glucose = [random.uniform(4, 14) for _ in range(len(dates))]
#     ketones = [random.uniform(0.1, 4.0) for _ in range(len(dates))]
#     return pd.DataFrame({'datetime': dates, 'glucose': glucose, 'ketones': ketones})

# # Calculate GKI
# def calculate_gki(glucose, ketones):
#     return glucose / (ketones)

# # Create graphs
# def create_graphs(data):
#     fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
#                         subplot_titles=("Glucose Levels", "Ketone Levels", "Glucose-Ketone Index (GKI)"))
    
#     fig.add_trace(go.Scatter(x=data['datetime'], y=data['glucose'], mode='lines', name='Glucose'),
#                   row=1, col=1)
#     fig.add_trace(go.Scatter(x=data['datetime'], y=data['ketones'], mode='lines', name='Ketones'),
#                   row=2, col=1)
    
#     gki = calculate_gki(data['glucose'], data['ketones'])
#     fig.add_trace(go.Scatter(x=data['datetime'], y=gki, mode='lines', name='GKI'),
#                   row=3, col=1)
    
#     fig.update_layout(height=900, title_text="Glucose, Ketone, and GKI Monitoring")
#     return fig


# Simulating realistic glucose and ketone data
def get_monitor_data(days=1):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days*288, freq='5T')  # 288 entries per day (5-minute intervals)
    
    # Simulate glucose: normal levels with spikes after meals (3 main meals a day)
    glucose = []
    for i in range(len(dates)):
        hour = dates[i].hour
        if 6 <= hour < 8 or 12 <= hour < 14 or 18 <= hour < 20:  # Around breakfast, lunch, dinner
            glucose.append(random.uniform(7, 10))  # Post-meal spike
        else:
            glucose.append(random.uniform(4, 6))  # Normal fasting levels
    
    # Simulate ketones: steady baseline with gradual increases during fasting
    ketones = []
    for i in range(len(dates)):
        hour = dates[i].hour
        if 6 <= hour < 8 or 12 <= hour < 14 or 18 <= hour < 20:  # Meals suppress ketones
            ketones.append(random.uniform(0.1, 0.5))
        else:
            ketones.append(random.uniform(0.5, 2.0))  # Higher ketones during fasting
    
    return pd.DataFrame({'datetime': dates, 'glucose': glucose, 'ketones': ketones})

# Calculate GKI
def calculate_gki(glucose, ketones):
    return glucose / (ketones)

# Create graphs
def create_graphs(data):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Glucose Levels", "Ketone Levels", "Glucose-Ketone Index (GKI)"))
    
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['glucose'], mode='lines', name='Glucose'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data['datetime'], y=data['ketones'], mode='lines', name='Ketones'),
                  row=2, col=1)
    
    gki = calculate_gki(data['glucose'], data['ketones'])
    fig.add_trace(go.Scatter(x=data['datetime'], y=gki, mode='lines', name='GKI'),
                  row=3, col=1)
    
    fig.update_layout(height=900, title_text="Glucose, Ketone, and GKI Monitoring")
    return fig


def main():
    st.set_page_config(page_title="VitaTraxx: Your AI Health Guru", page_icon="🥗")

    st.title("VitaTraxx: Your AI Health Guru")
    st.write("This is a simple app using Streamlit.")

    if 'food_list' not in st.session_state:
        st.session_state.food_list = []
    if 'orders' not in st.session_state:
        st.session_state.orders = {}

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Speak Food Item"):
            st.markdown('<div class="blinking-mic">🎙️</div>', unsafe_allow_html=True)
            with st.spinner("Listening..."):
                food = speech_to_text()
                if food:
                    st.write(f"You said: {food}")
                    nutrition_info = get_nutrition_info(food)
                    if nutrition_info:
                        log_food(food, nutrition_info)
                        st.session_state.food_list.append(food)
                        st.success(f"Logged {food} to your daily intake.")
                    else:
                        st.error("Couldn't find nutrition information for this food.")

    with col2:
        if st.button("Get Nutrition Advice"):
            daily_totals = get_daily_totals()
            weekly_totals = get_cumulative_totals(7)
            scores = calculate_nutrient_scores(daily_totals)
            advice = get_nutrition_advice(daily_totals, weekly_totals, scores)
            st.write("AI Coach says:")
            st.write(advice)
            
            st.markdown('<div class="speaking-android">🤖</div>', unsafe_allow_html=True)
            with st.spinner("Generating speech..."):
                speech_file = text_to_speech(advice)
                autoplay_audio(speech_file)

    st.subheader("Today's Food List")
    for food in st.session_state.food_list:
        st.write(f"• {food}")

    st.subheader("Daily Nutrition Progress")
    daily_totals = get_daily_totals()
    create_progress_bars(daily_totals)

    st.subheader("Cumulative Nutrition Progress (Last 7 Days)")
    cumulative_totals = get_cumulative_totals(7)
    create_progress_bars(cumulative_totals)

    st.subheader("Nutrient Intake Overview")
    create_nutrient_plot(daily_totals)

    st.subheader("Drill-down Food Nutrient Contribution")
    food_items = get_food_items_from_db()
    selected_food = st.selectbox("Select a food item for detailed nutrient breakdown:", food_items)

    # Get nutrient data for selected food
    nutrient_data = get_nutrient_data(selected_food)

    if nutrient_data:
        # Display nutrient data
        st.subheader(f"Nutrient Composition for {selected_food}")
        st.dataframe(pd.DataFrame([nutrient_data]).T.rename(columns={0: 'Value'}))
        
        # Display drill-down graph
        st.plotly_chart(create_nutrient_drill_down(nutrient_data, selected_food))
    else:
        st.warning("No nutrient data available for the selected food item.")

    # Nutrient comparison across foods
    st.subheader("Nutrient Comparison Across Foods")
    selected_nutrient = st.selectbox("Select a nutrient to compare across foods:", list(nutrients.values()))

    # Get the key for the selected nutrient
    selected_nutrient_key = [k for k, v in nutrients.items() if v == selected_nutrient][0]

    # Collect data for all food items
    comparison_data = []
    for food in food_items:
        food_data = get_nutrient_data(food)
        if food_data and food_data.get(selected_nutrient_key) is not None:
            comparison_data.append((food, food_data[selected_nutrient_key]))

    if comparison_data:
        fig = go.Figure(data=[
            go.Bar(name=food, x=[selected_nutrient], y=[value])
            for food, value in comparison_data
        ])

        fig.update_layout(
            title=f"{selected_nutrient} Content Across Foods",
            xaxis_title="Nutrient",
            yaxis_title="Content",
            barmode='group'
        )

        st.plotly_chart(fig)
    else:
        st.warning(f"No data available for {selected_nutrient} across foods.")

    st.subheader("Nutrient Scores")
    scores = calculate_nutrient_scores(daily_totals)
    st.write(f"Nutrient Score: {scores['nutrient_score']:.2f}%")
    st.write(f"Antinutrient Score: {scores['antinutrient_score']:.2f}%")
    st.write(f"Total Food Score: {scores['total_score']:.2f}%")

    st.subheader("Metabolic Correlation to Consumed Food Items")
    data = get_monitor_data()
    st.plotly_chart(create_graphs(data))
    food_items = ['Apple', 'Chicken Breast', 'Brown Rice', 'Spinach']
    correlations = [random.uniform(-1, 1) for _ in range(len(food_items))]
    correlation_df = pd.DataFrame({'Food Item': food_items, 'Correlation': correlations})
    st.table(correlation_df)

    st.subheader("Optimum Foods and InstaCart")
    missing_nutrients = get_missing_nutrients(daily_totals)
    #optimum_foods = get_optimum_foods(missing_nutrients)
    optimum_foods = get_recommended_foods(missing_nutrients)
    instacart = generate_instacart(optimum_foods)

    for store, items in instacart.items():
        if st.button(f"Checkout from {store}"):
            st.session_state.orders[store] = {"items": items, "status": "Order placed"}
            st.success(f"Order placed with {store}!")
    st.write(f"{store}: {', '.join(items)}")

    # Mock delivery updates
    for store, order in st.session_state.orders.items():
        st.write(f"{store} order status: {order['status']}")
    if st.button(f"Update {store} order status"):
        order['status'] = mock_delivery_update()
        st.success(f"{store} order status updated!")

    st.subheader("Food Recommendations")
    missing_nutrients = get_missing_nutrients(daily_totals)
    recommended_foods = get_recommended_foods(missing_nutrients)
    st.write("Recommended foods to improve your nutrition:")
    for food in recommended_foods:
        st.write(f"• {food}")

    st.subheader("Generate Recipe")
    if st.button("Generate Recipe from InstaCart Items"):
        all_ingredients = [item for items in instacart.values() for item in items]
        recipe = generate_recipe(all_ingredients)
        st.write(recipe)

# Add the call to main() here
if __name__ == "__main__":
    main()
    
# import numpy as np
# import streamlit as st
# from screens import home, generate_story, generate_noise, sleep_statistics, ai_recommendations
# from screens import device_integration, today_statistics, rag_screen, deep_seek, image_typewriter
# from screens import claude

# st.set_page_config(page_title="DreamWeaver AI", page_icon='💤', layout='wide')

# PAGES = {
#     "Home": home,
#     "Today's Sleep Statistics": today_statistics,
#     "Generate Story": generate_story,
#     "Generate Noise": generate_noise,
#     "Insights": sleep_statistics,
#     "AI Recommendations": ai_recommendations,
#     "Devices": device_integration,
#     "Retrieval-Augmented Generation": rag_screen,
#     "Image TypeWriter": image_typewriter,
#     "DeepSeek": deep_seek,
#     "Claude": claude,
# }

# st.sidebar.title('Navigation')
# selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# page = PAGES[selection]
# page.app()
