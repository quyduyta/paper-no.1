#!/usr/bin/env python

import wbdata
import pandas

wbdata.search_indicators("current account")
wbdata.search_indicators("unemployment")
indicators = {'BN.CAB.XOKA.GD.ZS': 'current account balance'}
df = wbdata.get_dataframe(indicators, country=countries, convert_date=False)
df.describe()

indicators = {'SL.UEM.TOTL.NE.ZS': 'unemployment'}
df = wbdata.get_dataframe(indicators, country=countries, convert_date=False)
df.describe()

countries = [i['id'] for i in wbdata.get_country(incomelevel="LMY", display=False)]
indicators = {"SL.UEM.TOTL.NE.ZS": "unemployment", "BN.CAB.XOKA.GD.ZS": "cab"}
df = wbdata.get_dataframe(indicators, country=countries, convert_date=True)
df.describe()
df = df.dropna()

df.cab.corr(df.unemployment)

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

countries = ["BEL", "EST", "HUN", "IRL", "VNM", "MLT"]
indicators = {"BN.CAB.XOKA.GD.ZS": "cab", "SL.UEM.TOTL.NE.ZS": "unemployment", "FI.RES.XGLD.CD": "reserve"}
df = wbdata.get_dataframe(indicators, country=countries, convert_date=True)
df.describe()
df = df.dropna()
mod = smf.ols('cab ~ cab.shift(-1) + np.log(reserve.shift(-1)) + unemployment.shift(-1)', data = df).fit()
print(mod.summary())

indicators = {"BN.CAB.XOKA.GD.ZS": "cab", "SL.UEM.TOTL.NE.ZS": "unemployment", "FI.RES.XGLD.CD": "reserve"}
df = wbdata.get_dataframe(indicators, country="all", convert_date=True)
mod = smf.ols('cab ~ cab.shift(-1) + np.log(reserve.shift(-1)) + unemployment.shift(-1)', data = df).fit()

countries = ["BEL", "EST", "HUN", "IRL", "VNM", "MLT"]
indicators = {"BN.CAB.XOKA.GD.ZS": "cab", "NY.GDP.PCAP.KD.ZG": "gdpgrowth", "FI.RES.XGLD.CD": "reserve"}
df = wbdata.get_dataframe(indicators, country=countries, convert_date=True)
mod = smf.ols('cab ~ cab.shift(-1) + np.log(reserve.shift(-1)) + gdpgrowth.shift(-1)', data = df).fit()
print(mod.summary())

indicators = {"BN.CAB.XOKA.GD.ZS": "cab", "NY.GDP.PCAP.KD.ZG": "gdpgrowth", "FI.RES.XGLD.CD": "reserve"}
df = wbdata.get_dataframe(indicators, country="all", convert_date=True)
mod = smf.ols('cab ~ cab.shift(-1) + np.log(reserve.shift(-1)) + gdpgrowth.shift(-1)', data = df).fit()
print(mod.summary())

indicators = {"BN.CAB.XOKA.GD.ZS": "cab", "NY.GDP.PCAP.KD.ZG": "gdpgrowth", "FI.RES.XGLD.CD": "reserve", "NE.TRD.GNFS.ZS": "openness", "TX.VAL.FUEL.ZS.UN": "fuelexports", "GC.DOD.TOTL.GD.ZS": "debt"}
df = wbdata.get_dataframe(indicators, country="all", convert_date=True)
mod = smf.ols('cab ~ cab.shift(-1) + np.log(reserve.shift(-1)) + gdpgrowth.shift(-1)+ openness.shift(-1)+ fuelexports.shift(-1)+ debt.shift(-1)', data = df).fit()
print(mod.summary())

*** 
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

indicators = {"BN.CAB.XOKA.GD.ZS": "cab", "SL.UEM.TOTL.NE.ZS": "unemployment", "FI.RES.XGLD.CD": "reserve", "NY.GDP.PCAP.KD.ZG": "gdpgrowth", "PX.REX.REER": "REER", "NE.TRD.GNFS.ZS": "openness", "TX.VAL.FUEL.ZS.UN": "fuelexports", "GC.DOD.TOTL.GD.ZS": "debt"}
df = wbdata.get_dataframe(indicators, country="all", convert_date=True)
df.describe()
mod = smf.ols('cab ~ cab.shift(-1) + unemployment.shift(-1) + np.log(reserve.shift(-1)) + gdpgrowth.shift(-1)+ REER.shift(-1) + openness.shift(-1)+ fuelexports.shift(-1)+ debt.shift(-1)', data = df).fit()
print(mod.summary())

indicators = {"BN.CAB.XOKA.GD.ZS": "cab", "SL.UEM.TOTL.NE.ZS": "unemployment", "FI.RES.XGLD.CD": "reserve", "NY.GDP.PCAP.KD.ZG": "gdpgrowth", "PX.REX.REER": "REER", "NE.TRD.GNFS.ZS": "openness", "TX.VAL.FUEL.ZS.UN": "fuelexports"}
df = wbdata.get_dataframe(indicators, country="all", convert_date=True)
mod = smf.ols('cab ~ cab.shift(-1) + unemployment.shift(-1) + np.log(reserve.shift(-1)) + gdpgrowth.shift(-1)+ REER.shift(-1) + openness.shift(-1)+ fuelexports.shift(-1)', data = df).fit()
print(mod.summary())
*result significant at 10%

indicators = {"SL.UEM.TOTL.NE.ZS": "unemployment", "BN.CAB.XOKA.GD.ZS": "cab", "FP.CPI.TOTL.ZG": "CPI", "NY.GDP.PCAP.KD.ZG": "gdpgrowth", "FI.RES.XGLD.CD": "reserve", "PX.REX.REER": "REER", "TX.VAL.FUEL.ZS.UN": "fuelexports", "GC.DOD.TOTL.GD.ZS": "debt", "SP.POP.GROW": "popgrowth"}
df = wbdata.get_dataframe(indicators, country="all", convert_date=True)
df.describe()
mod = smf.ols('unemployment ~ unemployment.shift(-1) + cab.shift(-1) + CPI.shift(-1) + gdpgrowth.shift(-1)+ REER.shift(-1) + fuelexports.shift(-1)', data = df).fit()
print(mod.summary())
*result significant at 1%

mod = smf.ols('unemployment ~ unemployment.shift(-1) + cab.shift(-1) + CPI.shift(-1) + gdpgrowth.shift(-1)+ fuelexports.shift(-1) + debt.shift(-1) + popgrowth.shift(-1)', data = df).fit()
print(mod.summary())
*result significant at 1%

indicators = {"SL.UEM.TOTL.NE.ZS": "unemployment", "BN.CAB.XOKA.GD.ZS": "cab", "FP.CPI.TOTL.ZG": "CPI", "NY.GDP.PCAP.KD.ZG": "gdpgrowth", "FI.RES.XGLD.CD": "reserve", "PX.REX.REER": "REER", "TX.VAL.FUEL.ZS.UN": "fuelexports", "GC.DOD.TOTL.GD.ZS": "debt", "SP.POP.GROW": "popgrowth", "SL.TLF.TOTL.IN": "laborforce", "SE.ADT.LITR.ZS": "literacy", "SE.XPD.TOTL.GD.ZS": "spendingEdu"}
df = wbdata.get_dataframe(indicators, country="all", convert_date=True)
mod = smf.ols('unemployment ~ unemployment.shift(-1) + cab.shift(-1) + CPI.shift(-1) + gdpgrowth.shift(-1)+ fuelexports.shift(-1) + debt.shift(-1) + popgrowth.shift(-1) + np.log(laborforce.shift(-1))', data = df).fit()
*result significant at 1%
