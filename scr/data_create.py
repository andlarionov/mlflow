import pandas as pd

# Открываем файл
df = pd.read_csv('datasets/shopping_trends.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

df = df[['age', 'gender', 'category', 'size', 'subscription_status', 'discount_applied']]

# Сохраняем данные в формате CSV
df.to_csv('datasets/df.csv', index=False)