import pandas as pd               # จัดการข้อมูลตาราง (DataFrame)
import numpy as np                # คำนวณตัวเลข
import matplotlib.pyplot as plt   # วาดกราฟ
import requests                   # ดึงข้อมูลจาก API
import json                       # แปลงข้อความ JSON เป็น object ของ Python
from pathlib import Path          # จัดการ path ไฟล์

# ให้กราฟแสดงใต้เซลล์ใน Jupyter
# %matplotlib inline

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)
print("Libraries loaded.") 

# ====================================

df = pd.read_csv("Export.csv", skiprows=1)  # อ่านข้อมูลจากไฟล์ CSV
print(df.shape)  # แสดงขนาดของ DataFrame
print(df.info())   # แสดงข้อมูลเกี่ยวกับ DataFrame
print(df.head())   # แสดงข้อมูล 5 แถวแรกของ DataFrame 

df.describe(include="all")  # แสดงสถิติพื้นฐานของข้อมูลใน DataFrame

print("Missing : \n", df.isna().sum())  # แสดงจำนวนค่าที่หายไปในแต่ละคอลัมน์
print("Missing Percentage : \n", (df.isna().mean() * 100).round(2))  # แสดงเปอร์เซ็นต์ของค่าที่หายไปในแต่ละคอลัมน์
print("Unique Values : \n", df.nunique())  # แสดงจำนวนค่าที่ไม่ซ้ำกันในแต่ละคอลัมน์

print("Duplicate rows:", df.duplicated().sum())  # แสดงจำนวนแถวที่ซ้ำกัน

# ====================================

d2 = df.drop_duplicates()  # ลบแถวที่ซ้ำกันออกจาก DataFrame
d2.reset_index(drop=True)  # รีเซ็ตดัชนีของ DataFrame หลังจากลบแถวที่ซ้ำกัน
print("\nAfter dropping duplicates:", d2.shape)  # แสดงขนาดของ DataFrame

# ====================================

df = df.fillna({
    "D/T": 0,
    "Wind Direction": 0,
    "Wind Speed": 0,
    "Temperature": 0,
    "Relative Humidity": 0,
    "PM 2.5": 0,
    "Atmospheric Pressure": 0,
    "Sensor 1": 0,
    "Sensor 2": 0,      
    "Sensor 3": 0,
    "Sensor 4": 0,
    "Sensor 5": 0,
    "Sensor 6": 0,
    "Sensor 7": 0,
    "Sensor 8": 0,
    "Smell Prediction": "Unknown"
}) # เติมค่า NaN ในแต่ละคอลัมน์ด้วยค่าที่กำหนด (0 สำหรับตัวเลข และ "Unknown" สำหรับข้อความ)
# (check missing values after filling)
# print(df)

# ====================================

# (Check data types before conversion)
# print("before:", df.dtypes.tolist())  # แสดงประเภทข้อมูลของแต่ละคอลัมน์ก่อนการแปลง

df["Time"] = pd.to_datetime(df["Time"])  # แปลงคอลัมน์ "Time" เป็นชนิด datetime 
#(Check data types after conversion)
# print("after:", df.dtypes.tolist())  # แสดงประเภทข้อมูลของแต่ละคอลัมน์หลังการแปลง

# ====================================

print("\nTotal missing:", df.isna().sum().sum())  # แสดงจำนวนค่าที่หายไปทั้งหมดใน DataFrame
print("Duplicate rows:", df.duplicated().sum())  # แสดงจำนวนแถวที่ซ้ำกันใน DataFrame

# ====================================

# สร้างคอลัมน์ใหม่โดยดึงข้อมูลจากคอลัมน์ "Time"
df["Month"] = df["Time"].dt.month  # สร้างคอลัมน์ "Month" โดยดึงเดือนจากคอลัมน์ "Time"
df["Day"] = df["Time"].dt.day  # สร้างคอลัมน์ "Day" โดยดึงวันจากคอลัมน์ "Time"
df["DateOnly"] = df["Time"].dt.date  # สร้างคอลัมน์ "DateOnly" โดยดึงวันที่จากคอลัมน์ "Time"
df["Hour"] = df["Time"].dt.hour  # สร้างคอลัมน์ "Hour" โดยดึงชั่วโมงจากคอลัมน์ "Time"
# แบ่งกลุ่มแต่ละหัวข้อ
df["DT_Group"] = pd.cut(
    df["D/T"].fillna(0),
    bins=[0, 15, 30, 45, 60],
    labels=["Very Low", "Low", "Medium", "High"],
    include_lowest=True
)
df["WindDir_Group"] = pd.cut(
    df["Wind Direction"].fillna(0),
    bins=[-1, 45, 90, 135, 180, 225, 270, 315, 360],
    labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
)
df["WindSpeed_Group"] = pd.cut(
    df["Wind Speed"].fillna(0),
    bins=[-1, 1, 5, 10, 20, 40],
    labels=["Calm", "Breeze", "Moderate", "Strong", "Storm"]
) 
# (check new columns)
# print(df.head())  # แสดงข้อมูล 5 แถวแรกของ DataFrame หลังจากเพิ่มคอลัมน์ใหม่

# ====================================

df[["D/T", "Wind Direction", "Wind Speed", "Month", "Day", "DateOnly", "Hour"]].describe()  # แสดงสถิติพื้นฐานของคอลัมน์ที่เกี่ยวข้องกับสภาพอากาศ  
df[["D/T", "Wind Direction", "Wind Speed", "Month", "Day", "DateOnly", "Hour", "DT_Group", "WindDir_Group", "WindSpeed_Group"]].head() # แสดงข้อมูล 5 แถวแรกของคอลัมน์ที่เกี่ยวข้องกับสภาพอากาศและกลุ่มที่สร้างขึ้น
print(df[["D/T", "Wind Direction", "Wind Speed", "Month", "Day", "DateOnly", "Hour", "DT_Group", "WindDir_Group", "WindSpeed_Group"]].head())  # แสดงข้อมูล 5 แถวแรกของคอลัมน์ที่เกี่ยวข้องกับสภาพอากาศและกลุ่มที่สร้างขึ้น
print(df[["D/T", "Wind Direction", "Wind Speed", "Month", "Day", "DateOnly", "Hour"]].describe())  # แสดงสถิติพื้นฐานของคอลัมน์ที่เกี่ยวข้องกับสภาพอากาศ (ใช้ display เพื่อแสดงผลใน Jupyter Notebook)

# ====================================

df[(df["D/T"] > 30) & (df["Wind Speed"] > 5)]   # แสดงแถวที่มีค่า D/T มากกว่า 30 และ Wind Speed มากกว่า 5
df[(df["DT_Group"].isin(["Medium", "High"])) & (df["WindSpeed_Group"].isin(["Moderate", "Strong", "Storm"]))]  # แสดงแถวที่มีค่า DT_Group เป็น "Medium" หรือ "High" และ WindSpeed_Group เป็น "Moderate", "Strong", หรือ "Storm"
print(df[(df["D/T"] > 30) & (df["Wind Speed"] > 5)])  # แสดงแถวที่มีค่า D/T มากกว่า 30 และ Wind Speed มากกว่า 5
print(df[(df["DT_Group"].isin(["Medium", "High"])) & (df["WindSpeed_Group"].isin(["Moderate", "Strong", "Storm"]))])  # แสดงแถวที่มีค่า DT_Group เป็น "Medium" หรือ "High" และ WindSpeed_Group เป็น "Moderate", "Strong", หรือ "Storm"

# ====================================3.4

