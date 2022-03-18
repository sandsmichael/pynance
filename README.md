# portrebopaly
portfolio rebalancing, optimization, and analysis

data sources:

* yfinance                          |   https://github.com/ranaroussi/yfinance
* Sharadar Core US Equities Bundle  |   https://data.nasdaq.com/databases/SFA/data  |   nasdaq data link (quandl)

*collection* --> aws ec2 | lambda functions --> daily pulls from above vendors
*storage* --> aws rds postgreesql 



portrebopaly/


model/

    attribution/
        risk & return performance attribution

    backtest/
        equity investment strategy backtesting 
    
    workflow/
        integrated workflow of various analytical tools included in the project for different products (i.e. equity or fixed income securities)

    finance/
        pricing fixed income instruments

    optimization/
        portfolio optimization using my forked version of PyPortfolioOpt

    performance/
        calculating performance
    
    rebalance/
        inteligent portfolio rebalancing 

    sec_nlp/
        summarization and key word extraction from sec corporate filings
    
    time_series/
        time series analysis

    flows/
        institutional v retail activity

    options/
        implied volitilities & put to call spreads

    screens/




view/

    gui/

        interface has three "starting points"

        1. Single Stock Analysis
            * workflows
            * technicals and fundamentals
            * sec filing nlp
            * dcf
        2. Portfolio Analysis
            * optimization
            * rebalance
        3. Market/Sector Analysis
            * screens
            * backtests




controller/








