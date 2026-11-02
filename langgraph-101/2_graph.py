'''
    START
      |
      v
 [load_application]
      |
  +---+---+----------------+          THREE CHECKS IN PARALLEL
  v       v                v
[income_check] [credit_history_check] [debt_check]
  |       |                |
  +---+---+----------------+
      v
  [aggregate]                         fan-in: combine into one score
      |
   +--+--+
 high    ok                           conditional edge
   |      |
   v      v
[send_to_underwriter] [auto_approve]
   |      |
   +--+---+
      v
     END

'''

APPLICATION = {
    "applicant_name": "Priya Sharma",
    "loan_amount": 1500000,
    "monthly_income": 85000,
    "credit_score": 690,
    "missed_payments_last_2_years": 2,
    "existing_emis": [
        {"lender": "HDFC Car Loan",    "monthly_emi": 18000},
        {"lender": "Bajaj Personal",   "monthly_emi": 9000},
        {"lender": "SBI Credit Card",  "monthly_emi": 6000},
    ],
}

'''
State must contain:

application_id
escalation_threshold
application
risk_score
decision
findings: a list that all three parallel checks write into
audit_log: a list that every node writes into

Run it twice
Strict bank: escalation_threshold = 4.0
Lenient bank: escalation_threshold = 7.5



'''