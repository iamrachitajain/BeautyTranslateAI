import streamlit as st
from PIL import Image
import pytesseract
from googletrans import Translator
from transformers import AutoTokenizer, pipeline
import os
# Custom CSS for Beautification
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f4f9;
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #ff6f61;
        text-align: center;
    }
    h2, h3 {
        color: #333;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        border-radius: 10px;
        padding: 8px 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff4c39;
        color: white;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
    }
    .stSelectbox {
        padding: 5px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Function to extract text from image using OCR (Tesseract)
def extract_text(image):
    return pytesseract.image_to_string(image)

# Function to translate the text to the selected language
def translate_text(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
terminators = tokenizer.eos_token_id

ingredient_model = pipeline(
    "text-generation",
    model='meta-llama/Llama-3.2-1B-Instruct',
    max_new_tokens=4096,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

# Define supported languages
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portugese": "pt",
    "Thai": "th",
    "Hindi": "hi"
    # Add more languages as needed
}

# Count the total number of supported languages
num_languages = len(languages)

# Streamlit App Layout
st.title("✨ Ingre Insight Genie ✨")
st.markdown("""
#### Welcome to **Ingre Insight Genie**! ✨  
This app helps you decode and translate the ingredients of beauty, skincare, and wellness products into simple, easy-to-understand terms. Whether you upload an image of the ingredient list or paste it directly, the app can break down the ingredients and explain their function, benefits, and possible concerns.  
Supports translation into 8 different languages to make it easy for anyone around the world to understand what’s in their products!
""")

st.markdown(f"##### Supported languages: {', '.join(languages.keys())}")
st.markdown("---")

# Use columns to place language selection and input options side by side
col1, col2 = st.columns(2)

with col1:
    st.markdown("##### Select Language for Translation")
    target_language = st.selectbox("", options=list(languages.keys()))

with col2:
    st.markdown("##### Choose input method")
    input_option = st.radio("", ["Paste Ingredients List", "Upload Image", ])

# 1. Option to upload an image of the ingredient list
if input_option == "Upload Image":
    st.markdown("##### Upload a beauty or wellness product image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # OCR: Extract text from the image
        st.markdown("##### Extracted Ingredients")
        with st.spinner("Extracting ingredients from the image..."):
            extracted_text = extract_text(image)
        st.success("Ingredients extracted successfully!")
        st.text_area("Extracted Text", value=extracted_text, height=150)

        # Translate extracted text
        if st.button("Translate from Image"):
            st.markdown(f"##### Translated Ingredients in {target_language}")
            with st.spinner("Translating..."):
                translated_text = translate_text(extracted_text, languages[target_language])
            st.success("Translation complete!")
            st.text_area(f"Translated Ingredients in {target_language}", value=translated_text, height=150)

# 2. Option to paste the ingredient list directly
elif input_option == "Paste Ingredients List":
    st.markdown("##### Paste the ingredient list below")
    ingredient_list = st.text_area("Enter or paste the ingredient/s here:", height=200, placeholder="E.g., Aqua, Glycerin, Sodium Chloride...")
    if ingredient_list:
        # Split ingredients by comma and clean up whitespace
        ingredients = [ing.strip() for ing in ingredient_list.split(",")]

        if st.button("Analyze Ingredients"):
            for ingredient in ingredients:
                with st.expander(f"Analysis for {ingredient}"):
                    with st.spinner(f"Analyzing {ingredient}..."):
                        prompt = f"""
                        Describe the ingredient {ingredient} \
                        in layman's terms in the language requested. Follow this structure:
                        About: Explain the ingredient in simple terms.
                        Harmful effects: List two possible harmful effects on health or the environment.
                        Answer only in the language {target_language}.
                        """
                        # Generate response
                        output = ingredient_model(prompt)
                        response_text = output[0]["generated_text"][len(prompt):].strip()
                        st.write(response_text)
                st.text_area(f"Ingredient breakdown in {target_language}", value=f"{response_text}", height=150)


