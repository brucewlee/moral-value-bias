import random
from datetime import datetime
import numpy as np 
from nutcracker.models import *

def model_selector(model_name):
    model_dict = {
        "openai_chatgpt" : OpenAI_ChatGPT(
            api_key=os.getenv("OPENAI_API_KEY", "default_api_key"),
            max_retries = 1000
        ), 
        "openai_chatgpt4" : OpenAI_ChatGPT4(
            api_key=os.getenv("OPENAI_API_KEY", "default_api_key"),
            max_retries = 1000
        ), 
        "openai_chatgpt4o" : OpenAI_ChatGPT4o(
            api_key=os.getenv("OPENAI_API_KEY", "default_api_key"),
            max_retries = 1000
        ), 
        "bedrock_claude3_opus" : Bedrock_Claude3_Opus(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY", "default_access_key"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY", "default_secret_key"),
            region_name="us-east-1",
            max_retries = 1000
        ),
        "bedrock_claude3_sonnet" : Bedrock_Claude3_Sonnet(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY", "default_access_key"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY", "default_secret_key"),
            region_name="us-east-1",
            max_retries = 1000
        ),
        "bedrock_claude3_haiku" : Bedrock_Claude3_Haiku(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY", "default_access_key"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY", "default_secret_key"),
            region_name="us-east-1",
            max_retries = 1000
        ),
        "bedrock_llama3_70b_inst" : Bedrock_LLaMA3_70B_Inst(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY", "default_access_key"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY", "default_secret_key"),
            region_name="us-east-1",
            max_retries = 1000
        ),
        "bedrock_llama3_8b_inst" : Bedrock_LLaMA3_8B_Inst(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY", "default_access_key"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY", "default_secret_key"),
            region_name="us-east-1",
            max_retries = 1000
        ),
        "cohere_commandrplus" : Cohere_CommandRPlus(
            api_key=os.getenv("COHERE_API_KEY", "default_api_key"),
            max_retries = 1000
        )
    }
    return model_dict.get(model_name)

def generate_persona_description(personas_per_question):    
    # Define variable categories
    sexes = ["Male", "Female"]#
    age_brackets = range(20, 81)#
    income_levels = range(1, 11)#
    have_children_options = ["Yes", "No"]#
    marriage_statuses = ["Married", "Living together as married", "Divorced", "Separated", "Widowed", "Single"]#
    education_levels = ["Early childhood education", "Primary education", "Lower secondary education", 
                        "Upper secondary education", "Post-secondary non-tertiary education", 
                        "Short-cycle tertiary education", "Bachelor or equivalent", 
                        "Master or equivalent", "Doctoral or equivalent"]#
    employment_statuses = ["full-time", "part-time", "not"]#
    occupation_groups = ["Professional and technical", "Higher administrative", "Clerical", "Sales", "Service", "Skilled worker", "Semi-skilled worker", "Unskilled worker", "Farm worker", "Farm proprietor, farm manager"]#
    ethnic_groups = ["White", "Black", "South Asian", "East Asian",  "Arabic", "Central Asian"]#
    religious_denominations = ["do not belong to a denomination", "Roman Catholic", "Protestant", "Orthodox", "Jew", "Muslim", "Hindu", "Buddhist"]#
    country_code_dict = {
        "8": "Albania",
        "20": "Andorra",
        "32": "Argentina",
        "51": "Armenia",
        "36": "Australia",
        "40": "Austria",
        "31": "Azerbaijan",
        "50": "Bangladesh",
        "112": "Belarus",
        "68": "Bolivia",
        "70": "Bosnia Herzegovina",
        "76": "Brazil",
        "100": "Bulgaria",
        "124": "Canada",
        "152": "Chile",
        "156": "China",
        "170": "Colombia",
        "191": "Croatia",
        "356": "India",
        "196": "Cyprus",
        "203": "Czechia",
        "208": "Denmark",
        "218": "Ecuador",
        "818": "Egypt",
        "233": "Estonia",
        "231": "Ethiopia",
        "246": "Finland",
        "250": "France",
        "268": "Georgia",
        "276": "Germany",
        "826": "Great Britain",
        "300": "Greece",
        "320": "Guatemala",
        "344": "Hong Kong SAR",
        "348": "Hungary",
        "352": "Iceland",
        "360": "Indonesia",
        "364": "Iran",
        "368": "Iraq",
        "380": "Italy",
        "392": "Japan",
        "400": "Jordan",
        "398": "Kazakhstan",
        "404": "Kenya",
        "417": "Kyrgyzstan",
        "428": "Latvia",
        "422": "Lebanon",
        "434": "Libya",
        "440": "Lithuania",
        "446": "Macao SAR",
        "458": "Malaysia",
        "462": "Maldives",
        "484": "Mexico",
        "496": "Mongolia",
        "499": "Montenegro",
        "504": "Morocco",
        "104": "Myanmar",
        "528": "Netherlands",
        "554": "New Zealand",
        "558": "Nicaragua",
        "566": "Nigeria",
        "807": "North Macedonia",
        "909": "Northern Ireland",
        "578": "Norway",
        "586": "Pakistan",
        "604": "Peru",
        "608": "Philippines",
        "616": "Poland",
        "620": "Portugal",
        "630": "Puerto Rico",
        "642": "Romania",
        "643": "Russia",
        "688": "Serbia",
        "702": "Singapore",
        "703": "Slovakia",
        "705": "Slovenia",
        "410": "South Korea",
        "724": "Spain",
        "752": "Sweden",
        "756": "Switzerland",
        "158": "Taiwan ROC",
        "762": "Tajikistan",
        "764": "Thailand",
        "788": "Tunisia",
        "792": "Turkey",
        "804": "Ukraine",
        "840": "United States",
        "858": "Uruguay",
        "862": "Venezuela",
        "704": "Vietnam",
        "716": "Zimbabwe"
    }#
    countries = list(country_code_dict.values())

    
    personas = []
    for _ in range(personas_per_question):
        sex = random.choice(sexes)
        age_bracket = random.choice(age_brackets)
        current_year = current_year = datetime.now().year
        birth_year = current_year - np.random.randint(age_bracket, age_bracket + 10)
        age = current_year - birth_year
        income_level = random.choice(income_levels)
        have_children = random.choice(have_children_options)
        marriage_status = random.choice(marriage_statuses)
        education_level = random.choice(education_levels)
        employment_status = random.choice(employment_statuses)
        occupation_group = random.choice(occupation_groups)
        ethnic_group = random.choice(ethnic_groups)
        religious_denomination = random.choice(religious_denominations)
        country_of_residence = random.choice(countries)
        country_of_origin = random.choice(countries)
        
        persona = f"You are a {sex} born in {birth_year}, which means that you are {age} years old. You were born in {country_of_origin} and you live in {country_of_residence}. Your income level is {income_level} out of 10. You are {'not married' if marriage_status == 'Single' else marriage_status.lower()}. You have {'no children' if have_children == 'No' else 'children'}. You have received {education_level.lower()} education. You are currently {employment_status} employed. {f'Your occupation group is {occupation_group}. ' if employment_status in ['full-time', 'part-time'] else ''}You are {ethnic_group}. You {'do not have a religion' if religious_denomination == 'do not belong to a denomination' else f'are {religious_denomination}'}."
        
        personas.append({
            'description': persona,
            'sex': sex,
            'age': age,
            'coutry_of_origin': country_of_origin,
            'country_of_residence': country_of_residence,
            'income_level': income_level,
            'marrital_status': marriage_status,
            'have_children': have_children,
            'education_level': education_level,
            'employment_status': employment_status,
            'occupation_group': occupation_group,
            'ethnic_group': ethnic_group,
            'religious_denomination': religious_denomination,
        })
    
    return personas