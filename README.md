# Laptop Price Prediction

### Problem Statement :
```bash
- Designed a web app that predicts the price of the laptop given the configurations.
- Scraped the laptops data from smartpix.com using selenium and BeautifulSoup package
- Use various ML/DL approach to find best model.
- Deploy model using flask library on cloud.
```

### Dataset :
```bash
We have scraped the data from smartpix.com.
The raw dataset contains 11 features and 1020 entries.`price` is the dependent feature 
and ramaining are the independent features.
```

### Attribute Information :
```bash
- model - brand & model name of the laptop.
- price - price of the laptop in rupee.
- score - score/ratings of each model.
- processor - types of the processor e.g. intel , ryzen wrt their generations.
- no_of_core - number of cores each processor have e.g.Dual Core,Quad core, Hexa core, Octa Core etc
- storage - type of storage e.g Hard disk, SSD in GB and TB
- ram - RAM of the particular model in GB.
- display - screen resolution of the laptop in pixels and touch type.
- os - Operating system type e.g Windows,Mac,Ubuntu,Android
- warranty - Warranty offered by companies in years.
```
### Web Scraping :
```bash
Data is scraped from smartpix.com website using selenium library. We try to extract the 
different features of the laptops such as:
- model name
- price
- score
- processor
- no of cores
- storage
- ram
- display
- os
- warranty
```

### Data Assesment & Cleaning
```bash
- Types of Assessment
There are 2 types of assessment styles
1. Manual - Looking through the data manually in google sheets
2. Programmatic - By using pandas functions such as info(), describe() or sample()

Scraped dataset have various quality issues and tidiness issues.Please refer 
data_accessing_and_cleaning.ipynb for more details.

```