import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_trace(df: pd.DataFrame, deg2rad=True) -> pd.DataFrame:
    """
    compute trace by method of minimal curve.
        'INC' - zenit
        'AZI' - azimut
        'MD' - length of path
    :param df: contain columns ['INC', 'AZI', 'MD'].
    :param deg2rad: if True values of 'INC', 'AZI' will convert to radians
    :return: same df with coordinates of trace ['X', 'Y', 'Z'] and addition data ['Radius', 'gamma', 'tangentX', 'tangentY', 'tangentZ']
    """
    if deg2rad:
        df['INC'] = np.deg2rad(df['INC'])
        df['AZI'] = np.deg2rad(df['AZI'])
    # касательная
    df['tangentX'] = np.sin(df['INC']) * np.cos(df['AZI'])
    df['tangentY'] = np.sin(df['INC']) * np.sin(df['AZI'])
    df['tangentZ'] = np.cos(df['INC'])
    # длинна кривой
    df['dMD'] = df['MD'] - df['MD'].shift(1).fillna(0)
    # угол дуги
    df['gamma'] = np.arccos(np.sin(df['INC'].shift(1)) * np.sin(df['INC']) * np.cos(df['AZI'] - df['AZI'].shift(1))
                            + np.cos(df['INC'].shift(1)) * np.cos(df['INC']))
    # радиус дуги
    df['Radius'] = df['dMD'] / df['gamma']
    df['RF'] = np.tan(0.5 * df['gamma']) * 2.0 / df['gamma']
    df['C'] = 0.5 * df['dMD'] * df['RF']
    # смещение
    df['dN'] = df['C'] * (df['tangentX'].shift(1) + df['tangentX'])
    df['dE'] = df['C'] * (df['tangentY'].shift(1) + df['tangentY'])
    df['dTVD'] = df['C'] * (df['tangentZ'].shift(1) + df['tangentZ'])
    # искомые координаты
    df['X'] = df['dN'].fillna(0).cumsum()
    df['Y'] = df['dE'].fillna(0).cumsum()
    df['Z'] = df['dTVD'].fillna(0).cumsum()
    #
    df.drop(['dN', 'dE', 'dTVD', 'RF', 'C', 'dMD'], axis=1, inplace=True)
    return df


def compute_circus(df: pd.DataFrame, add_radiuses=True) -> pd.DataFrame:
    """
    compute circus parameters by ['X', 'Y', 'Z', 'Radius', 'tangentX', 'tangentY', 'tangentZ']
    :param df:
    :param add_radiuses:
    :return: same df with ('v2c_X', 'v2c_Y', 'v2c_Z') - vector to center of circle from previous point, ('C_X', 'C_Y', 'C_Z') - center of circle
    """
    # Нормаль к плоскости диги
    tangent = df[['tangentX', 'tangentY', 'tangentZ']]
    normal = np.cross(tangent.shift(1), tangent)
    # Центр дуги
    v2c = np.cross(normal, tangent.shift(1))
    df[['v2c_X', 'v2c_Y', 'v2c_Z']] = v2c / np.linalg.norm(v2c, axis=1)[:, None]
    df['C_X'] = df['X'].shift(1) + df['Radius'] * df['v2c_X']
    df['C_Y'] = df['Y'].shift(1) + df['Radius'] * df['v2c_Y']
    df['C_Z'] = df['Z'].shift(1) + df['Radius'] * df['v2c_Z']
    if add_radiuses:
        dx = df['C_X'] - df['X']
        dy = df['C_Y'] - df['Y']
        dz = df['C_Z'] - df['Z']

        df['R1'] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        dx = df['C_X'] - df['X'].shift(1)
        dy = df['C_Y'] - df['Y'].shift(1)
        dz = df['C_Z'] - df['Z'].shift(1)

        df['R2'] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return df


def compute_subpoints(df: pd.DataFrame, dl=None, da=None, default_value=15):
    """
    :param default_value:
    :param dl:
    :param da:
    :param df:
    :return:
    """
    # compute parts count
    if dl is not None:
        df['parts_count'] = np.round((df['MD'] - df['MD'].shift(1)) / dl)
    elif da is not None:
        df['parts_count'] = np.round(df['gamma'] / da)
    else:
        df['parts_count'] = default_value

    result = pd.DataFrame()
    for i in range(len(df) - 1):
        a = df.iloc[i].copy()
        b = df.iloc[i + 1]
        if b['parts_count'] <= 1:
            result = pd.concat([pd.DataFrame([a]), result], ignore_index=True)
        else:
            #
            temp_df = pd.DataFrame()
            temp_df['MD'] = np.linspace(a['MD'], b['MD'], int(b['parts_count']))[1:-1]
            temp_df['INC'] = np.linspace(a['INC'], b['INC'], int(b['parts_count']))[1:-1]
            temp_df['AZI'] = np.linspace(a['AZI'], b['AZI'], int(b['parts_count']))[1:-1]
            temp_df['tangentX'] = np.linspace(a['tangentX'], b['tangentX'], int(b['parts_count']))[1:-1]
            temp_df['tangentY'] = np.linspace(a['tangentY'], b['tangentY'], int(b['parts_count']))[1:-1]
            temp_df['tangentZ'] = np.linspace(a['tangentZ'], b['tangentZ'], int(b['parts_count']))[1:-1]
            e2 = np.array([a['tangentX'], a['tangentY'], a['tangentZ']])
            e1 = np.array([-b['v2c_X'], -b['v2c_Y'], -b['v2c_Z']])
            center = np.array([b['C_X'], b['C_Y'], b['C_Z']])
            angels = np.linspace(0, b['gamma'], int(b['parts_count'])+1)[1:-1]
            subpoints = np.array([
                center + b['Radius'] * (np.cos(angel) * e1 + np.sin(angel) * e2)
                for angel in angels
            ])
            subpoints_df = pd.DataFrame(subpoints, columns=['X', 'Y', 'Z'])
            temp_df = temp_df.join(subpoints_df)
            temp_df['C_X'] = b['C_X']
            temp_df['C_Y'] = b['C_Y']
            temp_df['C_Z'] = b['C_Z']
            if i == 0:
                a['C_X'] = b['C_X']
                a['C_Y'] = b['C_Y']
                a['C_Z'] = b['C_Z']
            #
            result = pd.concat([pd.DataFrame([a]), result, temp_df], ignore_index=True)
    last_row = df.iloc[len(df) - 1]
    result = pd.concat([result, pd.DataFrame([last_row])], ignore_index=True)
    return result[['MD', 'INC', 'AZI', 'tangentX', 'tangentY', 'tangentZ', 'X', 'Y', 'Z', 'C_X', 'C_Y', 'C_Z']]


if __name__ == '__main__':
    df = pd.read_csv('./data/Datos_Wellbore-47.csv')
    df = df[df.index % 10 == 0]
    df1 = pd.DataFrame({
        'MD': [0, 1000],
        'INC': [0, 50.0 * np.pi / 180.0],
        'AZI': [0, 45.0 * np.pi / 180.0]
    })
    df = compute_trace(df, deg2rad=False)
    #
    df = compute_circus(df, add_radiuses=False)
    df = compute_subpoints(df, da=0.001)
    print(df.to_string(max_rows=10))