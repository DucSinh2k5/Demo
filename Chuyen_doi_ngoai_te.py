import requests
from bs4 import BeautifulSoup
import pandas as pd


url = "https://portal.vietcombank.com.vn/Usercontrols/TVPortal.TyGia/pXML.aspx"

resp = requests.get(url)
resp.encoding = "utf-8"
soup = BeautifulSoup(resp.text, "xml")
rows = []

def safe_float(x):
    return float(x.replace(",", "")) if x != "-" else None
for ex in soup.find_all("Exrate"):
   rows.append({
        "Mã ngoại tệ": ex["CurrencyCode"],
        "Tên ngoại tệ": ex["CurrencyName"],
        "Mua tiền mặt": safe_float(ex["Buy"]),
        "Mua chuyển khoản": safe_float(ex["Transfer"]),
        "Bán": safe_float(ex["Sell"])
    })

df = pd.DataFrame(rows)
# df = df.iloc[:, 1:]
# print(df.head())

df.to_csv("tygia_vietcombank.csv", index=False, encoding="utf-8-sig")

tien_vnd = float(input("Nhập số tiền VND: "))
ma_nt  =  input("Nhập mã ngoại tệ: ").upper()
ty_gia = df.loc[df["Mã ngoại tệ"] == ma_nt, "Bán"]

if not ty_gia.empty:
    tygia_value = float(ty_gia.values[0])
    so_ngoai_te = (tien_vnd  / tygia_value)
    print(f"{tien_vnd:,.0f} VND = {so_ngoai_te:,.2f} {ma_nt}")
else:
    print("Không tìm thấy mã ngoại tệ trong bảng tỷ giá.")