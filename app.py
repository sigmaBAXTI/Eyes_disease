import streamlit as slt
from fastai.vision.all import *
import pathlib
import plotly.express as ax
import platform
import pandas as pd

# plt = platform.system()
# if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# streamlit run .\app.py
# title
slt.title("Ko'z kasalliklarini turini aniqlab beradi : ")

matn = """Assalomu alaykum, bu tayyor matn.\nKaggle saytidan olingan dataset : https://www.kaggle.com/datasets/paultimothymooney/kermany2018/data\n
            Ko'z kasalliklari haqida quyidagi turdagi turlar mavjud:\n

            1. NORMAL (Normallik): Bu holatda ko'zlar normal bo'lib, kasallik belgilari yo'q. Bu odatiy, sog'liqni saqlash va normallashtirish bo'yicha yaxshi holatdir.\n

            2. CNV (Choroidal neovascularization):  Bu kasallik ko'zning o'rniga kirituvchi choroidal neovascularizationni ifodalaydi. Bu o'qshomli qandayliklar ko'zning orta qismida yuzaga keladi va ko'zning qorinidagi qandaylik tufayli ko'zning ishlashini buzadi.\n

            3. DME (Diabetic macular edema): Bu diabet diabetik retinopatiya bilan bog'liq kasallikdir. Bu holatda makulada suyuqroq to'sqinlarni olish bilan bog'liq boshqa komplikatsiyalar paydo bo'lishi mumkin. Bu holatda erkin asoslangan esiqlik, harrakatsizlik va ko'zni yo'qotish mumkin.\n

            4. DRUSEN (Drusen): Drusen makulada anormal protein o'chirishi yoki kalsiyum birikintilarini ifodalovchi qismlar hisoblanadi. Bu Ivolanmakula degan holatning oldini olish uchun, sarson rangli qopni ko'zde hisoblash mumkin.\n

            Bu kasalliklar profesional tibbiy maslahatga muhtojligini anglatadi. Agar sizda ko'z kasalligi haqida shubhalar yoki alomatlar mavjud bo'lsa, mutlaqo tug'ilgan ekspert yordamiga murojaat qiling.\n"""
slt.write('Hoyiha haqida :', matn)

slt.header("2023-yilda ko'zlarni tekshirish bo'yicha olingan statistika: ")
# Statistikali ma'lumotlarni yaratish
data = {
    'Kasallik': ['NORMAL', 'CNV', 'DME', 'DRUSEN'],
    'Soni': [190, 70, 100, 50]
}
df = pd.DataFrame(data)
# Statistikalar grafikini chizish
fig = ax.bar(df, x='Kasallik', y='Soni', color='Kasallik', labels={'Soni': 'Kasallik Soni'})
# Saytda ko'rsatish
slt.plotly_chart(fig)


# rasmni joylash - ya'ni tugma qo'shish
file = slt.file_uploader("Rasm yuklash", type=['png', 'jpg', 'jpeg', 'gif'])

if file:
    slt.image(file)
    # PIL convert
    img = PILImage.create(file)

    # model
    model = load_learner("Koz_model.pkl")

    # prediction
    pred, pred_id, probs = model.predict(img)

    if probs[pred_id] > 0.96:
        slt.success(f"Bashorat -> {pred}")
        slt.info(f"Ehtimollik  -> {probs[pred_id] * 100 :.1f} %")

        # plotting - ekranga ustun shaklida chiqarish
        fig = ax.bar(x=probs*100, y=model.dls.vocab)
        slt.plotly_chart(fig)
    else:
        slt.info("""Xatolik!!!\n
                    Iltimos, <<'png', 'jpg', 'jpeg', 'gif'>> formatdagi rasm kiriting\n
                    Yoki, Siz boshqa rasmni kiritgansiz.\n
                    Iltimos, Ko'zni tomografiya orqali tushurilgan sur'atini yuboring...""")