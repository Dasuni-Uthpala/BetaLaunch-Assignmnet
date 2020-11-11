import pandas as pd
import matplotlib.pyplot as plt
workbook = "sample_submission_my.xlsx"

df = pd.read_excel(workbook)

values = df[['Id','SalePrice']]

ax = values.plot.bar(x="Id",y="SalePrice",rot=0)
plt.show()
