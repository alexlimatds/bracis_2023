import dm_7_roles, dm_4_roles, dm_original

def get_data_manager(dataset_id):
    '''
    Returns a data manager from a provided ID. A data manager is an object 
    in charge of load a dataset and provide some data (e.g., labels) about it.
    '''
    if dataset_id == '7_roles':
        data_loader = dm_7_roles.SevenRolesDM()
    elif dataset_id == '4_roles':
        data_loader = dm_4_roles.FourRolesDM()
    elif dataset_id == 'original':
        data_loader = dm_original.OriginalDM()
    else:
        raise ValueError('Invalid dataset:', dataset_id)
    return data_loader
