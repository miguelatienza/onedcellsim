import pandas as pd

def df(parameter_dict):

    column_names = ['Name', 'Min', 'Default', 'Max', 'Type']
    columns = {'Name':[key for key in parameter_dict.keys()],
                'Min':[parameter_dict[key]['prior'][0] for key in parameter_dict.keys()],
                'Default':[parameter_dict[key]['prior'][1] for key in parameter_dict.keys()],
                'Max':[parameter_dict[key]['prior'][2] for key in parameter_dict.keys()],
                'Type':[parameter_dict[key]['type'] for key in parameter_dict.keys()],
    }
               
    df = pd.DataFrame(columns)
    #sort by type: variable first, then latent then constant
    order = ['variable', 'latent', 'constant']
    df = df.sort_values(by=['Type'], key=lambda x: x.map(dict(zip(order, range(len(order))))))
    return df
