from django.shortcuts import render

from django.shortcuts import render
from django.shortcuts import render
from matplotlib.font_manager import json_dump
import pandas as pd
import sys
import os
import json
import nasdaqdatalink
from pyparsing import line
from lib.calendar import Calendar
from lib.fixed_income.pricing.pricing import Pricing
from db.postgres import Postgres
from lib import numeric
import yfinance

cal = Calendar()
from dateutil.relativedelta import relativedelta
import datetime
from tabulate import tabulate





# Create your views here.
def price(request):

    pr  = Pricing()
    
    return render(request, 'price.html')

