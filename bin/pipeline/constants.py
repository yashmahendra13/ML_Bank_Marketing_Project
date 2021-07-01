from types import SimpleNamespace

feature_selection = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']

target = ['y']

categorical_features = ['contact','job','marital','education','default','housing','loan','contact','month','day_of_week','pdays','poutcome']

numerical_features = ['age','campaign', 'previous','emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
       'euribor3m', 'nr.employed']

