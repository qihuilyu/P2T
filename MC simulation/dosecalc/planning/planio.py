import numpy as np
import scipy.io

class OptimizedPlan():
    def __init__(self):
        self.name = None
        self.data = None
        self.dfcost = None
        self.cost= None
        self.date = None
        self.xpolish= None
        self.gtdose = None

    @classmethod
    def load_plans_from_file(cls, f):
        a = scipy.io.loadmat(f)['DoseInfo']

        names = np.squeeze(a['Name'])
        datas = np.squeeze(a['Data'])
        dfcosts = np.squeeze(a['DFCost'])
        costs = np.squeeze(a['Cost'])
        dates = np.squeeze(a['Date'])
        polishes = np.squeeze(a['xPolish'])
        gtdoses = np.squeeze(a['GTdose'])

        plans = {}
        for ii in range(len(names)):
            st = cls()
            st.name = names[ii][0]
            st.data = np.transpose(datas[ii], (2,0,1))
            st.dfcost = dfcosts[ii][0][0]
            st.costs= costs[ii][0][0]
            st.date = dates[ii][0]
            st.xpolish = np.squeeze(polishes[ii])
            st.gtdose= np.transpose(gtdoses[ii], (2,0,1))
            plans[st.name] = st

        return plans
