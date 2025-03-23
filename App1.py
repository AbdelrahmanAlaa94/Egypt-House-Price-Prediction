import gradio as gr
import numpy as np
import pandas as pd
import joblib  # If model is saved as a joblib file

# Load trained model (update the path if needed)
model = joblib.load("house_price_model.pkl")  # Ensure this file exists

# Define feature names based on training data
feature_columns = ["Location", "Area", "Bedroom", "Bathroom", "Floor", "View", "Facing", "Elevator", "Payment System"]

# Mapping dictionaries for categorical inputs
location_mapping = {
    "New Administrative Capital": 1, "First Settlement": 2, "Second Settlement": 3, "Third Settlement": 4,
    "Fourth Settlement": 5, "Fifth Settlement": 6, "Madinaty": 7, "El Rehab": 8, "Kattameya": 9,
    "Zahraa Nasr City": 10, "Nasr City": 11, "Maadi": 12, "Muqattam": 13, "Zahraa El Maadi": 14,
    "Sheraton": 15, "Heliopolis": 16, "Badr": 17, "El Obour": 18, "El Mostakbal": 19, "El Shorouk": 20,
    "El Salam": 21, "Gesr El Suez": 22, "El Marg": 23, "Ain Shams": 24, "Mostorod": 25, "Shubra El Kheima": 26,
    "Hadayek El Kobba": 27, "El Zawya El Hamra": 28, "Shubra": 29, "Rod El Farag": 30, "El Basatin": 31,
    "El Abbasiya": 32, "Sayeda Zeinab": 33, "Imbaba": 34, "El Azbakeya": 35, "El Fustat": 36, "Agouza": 37,
    "Boulak El Dakrour": 38, "El Mataria": 39, "El Nozha": 40, "El Waili": 41, "Abdeen": 42, "El Sharabiya": 43,
    "El Amireya": 44, "El Manial": 45, "Zamalek": 46, "El Mohandessin": 47, "Dokki": 48, "El Haram": 49,
    "El Warraq": 50, "El Omraniya": 51, "El Moneeb": 52, "Hadayek El Ahram": 53, "Sheikh Zayed": 54,
    "Hadayek October": 55, "New Sphinx": 56, "6th of October": 57, "El Faggala": 58, "Helwan": 59, "Faisal": 60
}

view_mapping = {"Garden": 1, "Street": 2, "Apartment": 3, "Apartment Building": 4} 
facing_mapping = {"North": 1, "South": 2}
elevator_mapping = {"Yes": 1, "No": 0}
payment_mapping = {"Cash": 0, "Installments": 1}

def predict_price(location, area, bedroom, bathroom, floor, view, facing, elevator, payment):
    # Convert categorical inputs to numerical values
    location = location_mapping.get(location, 0)
    view = view_mapping.get(view, 0)
    facing = facing_mapping.get(facing, 0)
    elevator = elevator_mapping.get(elevator, 0)
    payment = payment_mapping.get(payment, 0)
    
    # Prepare input data
    input_data = np.array([[location, area, bedroom, bathroom, floor, view, facing, elevator, payment]])
    input_df = pd.DataFrame(input_data, columns=feature_columns)
    
    # Predict price
    predicted_price = model.predict(input_df)[0]
    return f"💰 Predicted House Price: {predicted_price:,.2f} EGP"

# Gradio interface
demo = gr.Blocks(css="""
        body { background-color: #440154; }
        .gradio-container { 
            background: linear-gradient(135deg, #440154, #3b528b, #21918c, #5ec962, #fde725);
            padding: 20px;
            border-radius: 10px;
            color: white;
        }
        .gradio-container label {
            color: white;
        }
        .gradio-container .output-textbox {
            background-color: #fde725;
            color: black;
            font-weight: bold;
        }
        h1, p {
            text-align: center;
        }
    """)
with demo:
    gr.Markdown("# 🏠 House Price Prediction")
    gr.Markdown("Enter details to predict the house price in Egypt.")
    
    with gr.Column():
        inputs = [
            gr.Dropdown(choices=list(location_mapping.keys()), label="📍 Location"),
            gr.Slider(minimum=90, maximum=250, step=10, label="📏 Area (sqm)"),
            gr.Number(label="🛏 Bedroom", precision=0),
            gr.Number(label="🚻 Bathroom", precision=0),
            gr.Number(label="🏢 Floor", precision=0),
            gr.Radio(choices=list(view_mapping.keys()), label="🏩 View"),
            gr.Radio(choices=list(facing_mapping.keys()), label="🫏 Facing"),
            gr.Radio(choices=["Yes", "No"], label="🛇 Elevator"),
            gr.Radio(choices=["Cash", "Installments"], label="💰 Payment System")
        ]
        output = gr.Textbox(label="🔍 Predicted Price")
        with gr.Row():
            submit_button = gr.Button("Predict Price")
            clear_button = gr.Button("Clear")
        
        submit_button.click(predict_price, inputs=inputs, outputs=output)
        clear_button.click(lambda: "", inputs=[], outputs=output)

demo.launch()