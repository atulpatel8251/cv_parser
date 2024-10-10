import openai
import PyPDF2
import io
import json
import re
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image
import base64
from dateutil import parser
from dateutil.relativedelta import relativedelta
import logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Set Streamlit to use wide mode for the full width of the page
st.set_page_config(page_title="Resume Scanner", page_icon=":page_facing_up:",layout="wide")

# Load the logo image
logo_image = Image.open('assests/HD_Human_Resources_Banner.jpg')

# Optionally, you can resize only if the original size is too large
# For high-definition display, consider not resizing if the image is already suitable
resized_logo = logo_image.resize((1500, 300), Image.LANCZOS)  # Maintain quality during resizing

# Display the logo
st.image(resized_logo, use_column_width=True)

def extract_experience_from_cv(cv_text):
    # Regular expression to find date ranges
    date_pattern = r'(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{1,2})[-/\s](?:\d{4})\b)'
    date_matches = re.findall(date_pattern, cv_text)

    total_experience = relativedelta()
    experience_details = []

    for i in range(0, len(date_matches), 2):
        try:
            start_date_str = date_matches[i]
            end_date_str = date_matches[i+1] if i+1 < len(date_matches) else "Present"

            # Parse the start date
            start_date = parser.parse(start_date_str)

            # Check if the end date is 'Present' or an actual date
            if end_date_str.lower() == 'present':
                end_date = datetime.now()
            else:
                end_date = parser.parse(end_date_str)

            # Calculate the duration using relativedelta
            duration = relativedelta(end_date, start_date)
            total_experience += duration

            # Store the detailed experience
            experience_details.append({
                "start": start_date.strftime('%Y-%m'),
                "end": end_date.strftime('%Y-%m') if end_date_str.lower() != 'present' else end_date_str,
                "duration": f"{duration.years} years, {duration.months} months"
            })
        except ValueError as e:
            logging.error(f"Date parsing error: {e}")

    # Convert total experience to years and months
    total_years = total_experience.years + total_experience.months / 12

    return {
        "total_years": total_years,
        "experience_details": experience_details
    }
# Function to add background image from a local file
def add_bg_from_local(image_file,opacity=0.8):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()

    # Inject custom CSS for background
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})),url("data:assests/logo.jfif;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image path
add_bg_from_local('assests/OIP.jfif')  # Adjust path to your image file

# Add styled container for the title and description with a maximum width
st.markdown("""
    <div style="background-color: lightblue; padding: 20px; border-radius: 10px; text-align: center; 
                 max-width: 450px; margin: auto;">
        <p style="color: black; margin-left: 20px; margin-top: 20px; font-weight: bold;font-size: 40px;">CV Screening Portal</p>
        <p style="color: black; margin-left: 15px; margin-top: 20px; font-size: 25px;">AI based CV screening</p>
    </div>
""", unsafe_allow_html=True)

# Change background color using custom CSS
st.markdown(
    """
    <style>
    body {
               background-color: green;

    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hide Streamlit's default footer and customize H1 style
hide_streamlit_style = """
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with open('style.css') as f:
 st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) 
# Example: Add your image handling or other logic here
images = ['6MarkQ']

openai.api_key = st.secrets["secret_section"]["OPENAI_API_KEY"]  # Replace with your actual OpenAI API key
#openai.api_key = st.secrets["secret_section"]["OPENAI_API_KEY"]


# Function to extract text from PDF uploaded via Streamlit
def extract_text_from_uploaded_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF file.

    Args:
        uploaded_file: A file-like object containing the PDF.

    Returns:
        str: Extracted text from the PDF or an empty string if an error occurs.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        logging.error(f"Error reading PDF: {e}")  # Log the error
        return ""

# Function to download the PDF resume
def generate_download_button(cv_file):
    # Create a download button for each resume
    return st.download_button(
        label="Download Resume",
        data=cv_file.read(),  # The PDF content to be downloaded
        file_name=cv_file.name,  # The file name to be used in the download
        mime="application/pdf"
    )

# Function to use GenAI to extract criteria from job description
def use_genai_to_extract_criteria(jd_text):
    prompt = (
        "Extract and structure the following details from the job description: "
        "1. Education requirements "
        "2. Required experience "
        "3. Mandatory skills "
        "4. Certifications "
        "5. Desired skills (for brownie points). "
        "The job description is as follows:\n\n"
        f"{jd_text}\n\n"
        "Please provide the response as a JSON object. For example:\n"
        "{\"education\": \"Bachelor's Degree, Master's Degree\", "
        "\"experience\": \"5 years experience in data science\", "
        "\"skills\": \"Python, SQL, Machine Learning\", "
        "\"certifications\": \"AWS Certified, PMP\", "
        "\"desired_skills\": \"Deep Learning, NLP\"}"
    )
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.5
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            return content
        except json.JSONDecodeError:
            st.error("Failed to parse JSON from AI response. Here's the raw response:")
            st.write(content)
            return ""
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return ""
    
def calculate_skill_score(skill_scores):
    # Sum up all the skill scores
    total_score = sum(skill_scores.values())    
    return total_score
    
def extract_experience_from_cv(cv_text):
    """
    Extract total years of experience from the given CV text.
    
    Parameters:
    cv_text (str): The CV text from which to extract work experience dates.

    Returns:
    dict: A dictionary containing the total years of experience and a detailed breakdown of experiences.
    """
    # Regular expression to find date ranges
    date_pattern = r'(\w+\s\d{4})\s*-\s*(\w+\s\d{4}|Present|\d{4})'
    date_matches = re.findall(date_pattern, cv_text)

    total_years = 0
    experience_details = []

    for match in date_matches:
        start, end = match[0], match[1]

        try:
            # Parse the start date
            start_date = datetime.strptime(start, '%B %Y')

            # Handle if the end date is "Present"
            if end.lower() == 'present':
                end_date = datetime.now()  # Assume current date as end
            elif end.strip().isdigit():  # Case for only year provided
                end_date = datetime.strptime(end.strip(), '%Y')
            else:
                end_date = datetime.strptime(end, '%B %Y')

            # Calculate the duration in years and months
            years = end_date.year - start_date.year
            months = end_date.month - start_date.month
            if months < 0:
                years -= 1
                months += 12

            total_experience_in_years = years + months / 12.0  # Convert months to decimal years
            total_years += total_experience_in_years

            experience_details.append({
                "start": start_date.strftime('%Y-%m'),
                "end": end_date.strftime('%Y-%m'),
                "duration_years": total_experience_in_years
            })

        except ValueError as e:
            print(f"Error parsing dates: {e}")

    return {
        "total_years": round(total_years, 2),  # Return total experience rounded to 2 decimal places
        "experience_details": experience_details
    }

# Example usage:
cv_text = """
Software Developer at XYZ Corp
March 2015 - June 2019

Senior Developer at ABC Inc
July 2019 - Present
"""

experience_info = extract_experience_from_cv(cv_text)
print(experience_info)

def match_cv_with_criteria(cv_text, criteria_json):
    if not criteria_json:
        st.error("Criteria JSON is empty or invalid.")
        results = {'cv_text': cv_text}  # Add this line before returning results
        return results

    try:
        criteria = json.loads(criteria_json)

        # Extract total years of experience from the CV
        experience_info = extract_experience_from_cv(cv_text)
        total_years = experience_info["total_years"]

        prompt = (
            "Given the job description criteria and the candidate's CV text, "
            "please match the following: "
            "1. Which education qualifications from the job description are present in the CV? "
            "2. Which experiences from the job description are present in the CV? "
            "3. Which mandatory skills and certifications from the job description are present in the CV? "
            "4. Which desired skills from the job description are present in the CV? "
            "5. Identify missing qualifications, experiences, skills, or certifications from the CV. "
            "Also, assign a score (0 to 10) for each desired skill based on the extent of match. "
            "The job description criteria are as follows:\n\n"
            f"{criteria_json}\n\n"
            "The CV text is as follows:\n\n"
            f"{cv_text}\n\n"
            "Please provide the response in the following format: "
            "{\"matching_education\": [...], \n"
            "\"matching_experience\": [...], \n"
            "\"matching_skills\": [...], \n"
            "\"matching_certifications\": [...], \n"
            "\"skill_scores\": {\"skill_1\": 8, \"skill_2\": 6, ...}, \n"
            
        )

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.5
        )

        matching_results = response.choices[0].message.content.strip()

        results = json.loads(matching_results)

        # Check pass/fail conditions based on required education, experience, skills, and certifications
        pass_fail = "Pass"  # Default to pass

        if not results.get("matching_education"):
            pass_fail = "Fail"

        if not results.get("matching_experience"):
            pass_fail = "Fail"

        if not results.get("matching_skills"):
            pass_fail = "Fail"

        if criteria.get("certifications") and not results.get("matching_certifications"):
            pass_fail = "Fail"

        # Extract skill scores from results and calculate the overall skill score
        skill_scores = results.get("skill_scores", {})
        overall_skill_score = calculate_skill_score(skill_scores)

        # Add pass/fail result, overall skill score, and total years of experience to the output
        results["pass_or_fail"] = pass_fail
        results["skill_score"] = overall_skill_score  # Ensure 'skill_score' is added
        results["total_years_of_experience"] = total_years  # Add total years of experience

        return results

    except json.JSONDecodeError as e:
        st.error(f"Error parsing criteria JSON: {e}")
        return {}

    except json.JSONDecodeError as e:
        st.error(f"Error parsing criteria JSON: {e}")
        return {}
    
if "justifications" not in st.session_state:
    st.session_state.justifications = {}

# Function to justify skill scoring based on the candidate's resume

def get_skill_score_justification(candidate_name, skill, score, cv_text):
    if score == 0:
        prompt = (
            f"Briefly explain in 2-3 bullet points why the candidate's resume text '{cv_text}' does not demonstrate the "
            f"necessary skills for '{skill}', resulting in a score of {score}/10. Focus on clear, simple reasons why the "
            "candidate lacks this skill."
        )
    else:
        prompt = (
            f"Provide a concise justification in 2-3 bullet points explaining why the candidate's resume text '{cv_text}' "
            f"demonstrates a match for the skill '{skill}' with a score of {score}/10. Keep each bullet simple, focusing "
            "on clear reasoning in 7-8 words per line. Avoid overly technical terms."
        )

    # Using the 'openai.chat.completions.create' method
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,  # Adjust tokens to focus on concise answers
        temperature=0.7
    )

    # Access the explanation from the response
    explanation = response.choices[0].message.content.strip()

    # Store explanation in session state
    st.session_state.justifications.setdefault(candidate_name, {})[skill] = explanation

    return explanation

def display_pass_fail_verdict(results, cv_text):
    candidate_name = results['candidate']
    skill_scores = results.get("skill_scores", {})
    
    # Debugging: Print skill_scores to verify the data
    logging.debug(f"Skill scores for {candidate_name}: {skill_scores}")

    with st.container():
        st.markdown("<div style='padding: 10px; background-color: #f0f8ff; border-radius: 10px;'>", unsafe_allow_html=True)
        
        pass_fail = results.get("pass_or_fail", "Fail")
        if pass_fail == 'Pass':
            st.markdown(f"<h2 style='color: green;'>🟢 Final Result: PASS ✔️</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: red;'>🔴 Final Result: FAIL ❌</h2>", unsafe_allow_html=True)

        if skill_scores:
            for skill, score in skill_scores.items():
                st.markdown("<hr>", unsafe_allow_html=True)
                expander = st.expander(f"**{skill}**")
                with expander:
                    justification_key = f"justification_{candidate_name}_{skill}"
                    if justification_key not in st.session_state:
                        explanation = get_skill_score_justification(candidate_name, skill, score, cv_text)
                        st.session_state[justification_key] = explanation
                    
                    # Display justification
                    st.write(st.session_state[justification_key])

                # Display the score and progress bar consistently
                st.markdown(f"**Score**: {score}/10")
                st.progress(score / 10.0)

        overall_skill_score = round(results.get("skill_score", 0), 1)
        st.markdown(f"<h3>Overall Skill Score: <strong>{overall_skill_score:.1f}</strong> out of 50</h3>", unsafe_allow_html=True)

        if pass_fail == "Pass":
            st.success("The candidate has passed based on the job description criteria.")
        else:
            st.error("The candidate has failed to meet the job description criteria.")

        st.markdown("</div>", unsafe_allow_html=True)

# Function to display and rank candidates in a table
def display_candidates_table(candidates):
    if not candidates:
        st.info("No candidates to display.")
        return

    # Create DataFrame from candidates
    df = pd.DataFrame(candidates)

    # Ensure 'pass_or_fail' and 'skill_score' columns exist
    if 'pass_or_fail' not in df.columns or 'skill_score' not in df.columns or 'total_years_of_experience' not in df.columns or 'candidate' not in df.columns:
        st.error("Missing required columns in candidates data.")
        return

    # Sort by pass/fail first, then by skill score
    df['rank'] = df.apply(lambda row: (0 if row['pass_or_fail'] == 'Pass' else 1, -row['skill_score'], -row['total_years_of_experience']), axis=1)
    df = df.sort_values(by='rank').drop(columns=['rank'])  # Dropping 'rank' for display

    # Reorder columns to make 'candidate' the first column
    columns_order = ['candidate', 'pass_or_fail', 'skill_score', 'total_years_of_experience'] + [col for col in df.columns if col not in ['candidate', 'pass_or_fail', 'skill_score', 'total_years_of_experience']]
    df = df[columns_order]

    # Display the table with beautification
    st.markdown("## :trophy: Candidate Rankings")
    
    # Apply coloring function for pass/fail column
    def color_pass_fail(val):
        color = 'lightgreen' if val == 'Pass' else 'lightcoral'
        return f'background-color: {color}'

    # Use Styler for table display
    styled_df = df.style.applymap(color_pass_fail, subset=['pass_or_fail'])
    st.dataframe(styled_df)  # Display the DataFrame

# Add a styled box for the file uploaders
st.markdown("""
    <div style="background-color: lightblue; padding: 4px; border-radius: 5px; text-align: left; 
                 max-width: 380px;">
        <p style="color: black; margin-left: 60px; margin-top: 20px; font-weight: bold;font-size: 20px;">Upload Job Description (PDF)</p>
    </div>
""", unsafe_allow_html=True)

jd_file = st.file_uploader(" ", type="pdf")  # Note the space in the label to keep it blank

st.markdown("""
    <div style="background-color: lightblue; padding: 2px; border-radius: 5px; text-align: left; 
                 max-width: 400px;">
        <p style="color: black; margin-left: 55px; margin-top: 20px; font-weight: bold;font-size: 20px;">Upload Candidate Resumes (PDF)</p>
    </div>
""", unsafe_allow_html=True)

# File upload for candidate resumes
cv_files = st.file_uploader("", type="pdf", accept_multiple_files=True)


# Ensure criteria_json is initialized in session state
if 'criteria_json' not in st.session_state:
    st.session_state['criteria_json'] = None

# Button to extract criteria from job description
if st.button("Extract Criteria"):
    if jd_file:
        jd_text = extract_text_from_uploaded_pdf(jd_file)
        if jd_text:
            criteria_json = use_genai_to_extract_criteria(jd_text)
            if criteria_json:
                # Save criteria in session state
                st.session_state.criteria_json = criteria_json
                st.success("Job description criteria extracted successfully.")
            else:
                st.error("Failed to extract job description criteria.")
        else:
            st.error("The uploaded JD file appears to be empty.")
    else:
        st.error("Please upload a Job Description PDF.")

# Button to match candidates with job description criteria
if st.button("Match Candidates"):
    if cv_files:
        if st.session_state.criteria_json:
            candidates_results = []
            for cv_file in cv_files:
                cv_text = extract_text_from_uploaded_pdf(cv_file)
                if cv_text:
                    results = match_cv_with_criteria(cv_text, st.session_state.criteria_json)
                    if results:
                        results['candidate'] = cv_file.name  # Adding resume name to the results
                        candidates_results.append(results)
                else:
                    st.error(f"Failed to extract text from CV: {cv_file.name}")

            # Display candidates table first
            if candidates_results:
                display_candidates_table(candidates_results)
                
            # Then display individual matching details
            for result in candidates_results:
                st.markdown(f"### Matching Results for {result['candidate']}:")
                display_pass_fail_verdict(result, result['candidate'])
        else:
            st.error("Please extract job description criteria first.")
    else:
        st.error("Please upload at least one CV PDF.")


disclaimer_text = """
<div style="color: grey; margin-top: 90px; font-style: italic;">
    <strong>*Disclaimer: </strong>The validity of certifications must be verified manually. These results are AI-generated and this tool is under development, candidates should be thoroughly evaluated through technical interviews based on the information provided in their resumes. Any gaps in a candidate's profile should be addressed and justified.
</div>
"""

st.markdown(disclaimer_text, unsafe_allow_html=True)        
        
footer = """
    <style>
    body {
        margin: 0;
        padding-top: 70px;  /* Add padding to prevent content from being hidden behind the footer */
    }
    .footer {
        position: absolute;
        top: 80px;
        left: 0;
        width: 100%;
        background-color: #002F74;
        color: white;
        text-align: center;
        padding: 5px;
        font-weight: bold;
        z-index: 1000;  /* Ensure it is on top of other elements */
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .footer p {
        font-style: italic;
        font-size: 14px;
        margin: 0;
        flex: 1 1 50%;  /* Flex-grow, flex-shrink, flex-basis */
    }
    @media (max-width: 600px) {
        .footer p {
            flex-basis: 100%;
            text-align: center;
            padding-top: 10px;
        }
    }
    </style>
    <div class="footer">
        <p style="text-align: left;">Copyright © 2024 MPSeDC. All rights reserved.</p>
        <p style="text-align: right;">The responses provided on this website are AI-generated. User discretion is advised.</p>
    </div>
"""

st.markdown(footer, unsafe_allow_html=True)
