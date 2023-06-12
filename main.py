"""
API code
"""
from fastapi import FastAPI
from enum import Enum
from pandas import DataFrame
import numpy as np
import uvicorn
from pydantic import BaseModel
from tools.utils import process_data, cat_features, cols
from tools.model import inference
import joblib

# create app
app = FastAPI()

class Workclass(str, Enum):
    SG = "State-gov"
    SENI = "Self-emp-not-inc"
    PR = "Private"
    FG = "Federal-gov"
    LG= "Local-gov"
    SEI = "Self-emp-inc"
    WP = "Without-pay"

class Education(str, Enum):
    BA = "Bachelors"
    HS = "HS-grad"
    ELTH = "11th"
    MA = "Masters"
    NITH = "9th"
    SC = "Some-college"
    AA = "Assoc-acdm"
    SEEITN = "7th-8th"
    DOC = "Doctorate"
    AV = "Assoc-voc"
    PS = "Prof-school"
    FISITH = "5th-6th"
    TETH = "10th"
    PR = "Preschool"
    TWTH = "12th"
    ONFOTH = "1st-4th"

class MaritalStatus(str, Enum):
    NM = "Never-married"
    MCS = "Married-civ-spouse"
    DI = "Divorced"
    MSA = "Married-spouse-absent"
    SE = "Separated"
    MAS = "Married-AF-spouse"
    WI = "Widowed"

class Occupation(str, Enum):
    TS = "Tech-support"
    CR = "Craft-repair"
    OS = "Other-service"
    SA = "Sales"
    EM = "Exec-managerial"
    PS = "Prof-specialty"
    HC = "Handlers-cleaners"
    MOI = "Machine-op-inspct"
    AC = "Adm-clerical"
    FF = "Farming-fishing"
    TM = "Transport-moving"
    PHS = "Priv-house-serv"
    PS1 = "Protective-serv"
    AF = "Armed-Forces"

class Relationship(str, Enum):
    WI= "Wife"
    OC = "Own-child"
    HUS = "Husband"
    NIF = "Not-in-family"
    OR = "Other-relative"
    UN = "Unmarried"

class Race(str, Enum):
    WI = "White"
    API = "Asian-Pac-Islander"
    AIE = "Amer-Indian-Eskimo"
    O = "Other"
    B = "Black"

class Sex(str, Enum):
    F = "Female"
    M = "Male"

class NativeCountry(str, Enum):
    USA ="United-States"
    CU ="Cuba"
    JAM ="Jamaica"
    IND ="India"
    ME ="Mexico"
    PUE ="Puerto-Rico"
    HON ="Honduras"
    ENG ="England"
    CAN ="Canada"
    GER ="Germany"
    IRA ="Iran"
    PHI ="Philippines"
    POL ="Poland"
    COL ="Columbia"
    CAM ="Cambodia"
    THA ="Thailand"
    ECU ="Ecuador"
    LAO ="Laos"
    TAI ="Taiwan"
    HAI ="Haiti"
    POR ="Portugal"
    DOM ="Dominican-Republic"
    EL ="El-Salvador"
    FR ="France"
    GU ="Guatemala"
    IT ="Italy"
    CN ="China"
    SO ="South"
    JA = "Japan"
    YU = "Yugoslavia"
    PRE = "Peru"
    OUT = "Outlying-US(Guam-USVI-etc)"
    SC = "Scotland"
    TR = "Trinadad&Tobago"
    GR = "Greece"
    NI = "Nicaragua"
    VN = "Vietnam"
    HO = "Hong"
    IR = "Ireland"
    HU = "Hungary"
    HL = "Holand-Netherlands"

# POST Input Schema
class InformationInput(BaseModel):
    age: int
    workclass: Workclass
    fnlgt: int
    education: Education
    education_num: int
    marital_status: MaritalStatus
    occupation: Occupation
    relationship: Relationship
    race: Race
    sex: Sex
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: NativeCountry

    class Config:
        schema_extra = {
            "example": {
                "age": 50,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 83311,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 13,
                "native_country": "United-States"
            }
        }

# load models
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")

@app.get("/")
async def root():
    return {"greeting": "Hello, This is a test"}

# Prediction path
@app.post("/predict")
async def predict(input: InformationInput):

    input_data = np.array([[
        input.age,
        input.workclass,
        input.fnlgt,
        input.education,
        input.education_num,
        input.marital_status,
        input.occupation,
        input.relationship,
        input.race,
        input.sex,
        input.capital_gain,
        input.capital_loss,
        input.hours_per_week,
        input.native_country]])

    data = DataFrame(data=input_data, columns=cols)

    X, _, _, _ = process_data(
        data, categorical_features=cat_features, encoder=encoder, lb=lb, training=False)
    y = inference(model, X)
    pred = lb.inverse_transform(y)[0]

    return {"result": pred}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
