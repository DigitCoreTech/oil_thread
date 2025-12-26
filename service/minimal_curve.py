import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def compute_trace(df: pd.DataFrame, deg2rad: bool = True, min_angel: float = 0.5, *args, **kwargs) -> pd.DataFrame:
    """
    compute trace by method of minimal curve.
        'INC' - zenit
        'AZI' - azimut
        'MD' - length of path
    :param min_angel: min angel in same measure as computed from df
    :param df: contain columns ['INC', 'AZI', 'MD'].
    :param deg2rad: if True values of 'INC', 'AZI' will convert to radians
    :return: same df with coordinates of trace ['X', 'Y', 'Z'] and addition computed
    ['Radius', 'gamma', 'tangentX', 'tangentY', 'tangentZ']
    """

    if deg2rad:
        df['INC'] = np.deg2rad(df['INC'])
        df['AZI'] = np.deg2rad(df['AZI'])
        min_angel = np.deg2rad([min_angel])[0]
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
    df['RF'] = np.where(
        df['gamma'] <= min_angel,
        1.0,
        np.tan(0.5 * df['gamma']) * 2.0 / df['gamma']
    )
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


def compute_circus(df: pd.DataFrame) -> pd.DataFrame:
    """
    compute circus parameters by ['X', 'Y', 'Z', 'Radius', 'tangentX', 'tangentY', 'tangentZ']
    :param df:
    :return: same df with:
     ('v2c_X', 'v2c_Y', 'v2c_Z') - vector to center of circle from previous point
     ('C_X', 'C_Y', 'C_Z') - center of circle
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
    return df


def compute_subpoints(df: pd.DataFrame, dl: float = 10, da: float = None, default_value: int = 10, *args,
                      **kwargs) -> pd.DataFrame:
    """
    :param default_value:
    :param dl:
    :param da:
    :param df:
    :return:
    """
    # compute parts count
    if dl is not None:
        df['parts_count'] = np.round((df['MD'] - df['MD'].shift(1)) / dl) + 1
    elif da is not None:
        df['parts_count'] = np.round(df['gamma'] / da) + 1
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
            angels = np.linspace(0, b['gamma'], int(b['parts_count']) + 1)[1:-1]
            subpoints = np.array([
                center + b['Radius'] * (np.cos(angel) * e1 + np.sin(angel) * e2)
                for angel in angels
            ])
            subpoints_df = pd.DataFrame(subpoints, columns=['X', 'Y', 'Z'])
            temp_df = temp_df.join(subpoints_df)
            temp_df['C_X'] = b['C_X']
            temp_df['C_Y'] = b['C_Y']
            temp_df['C_Z'] = b['C_Z']
            temp_df['gamma'] = b['gamma']
            temp_df['Radius'] = b['Radius']
            if i == 0:
                a['C_X'] = b['C_X']
                a['C_Y'] = b['C_Y']
                a['C_Z'] = b['C_Z']
                a['gamma'] = b['gamma']
            #
            result = pd.concat([result, pd.DataFrame([a]), temp_df], ignore_index=True)
    last_row = df.iloc[len(df) - 1]
    result = pd.concat([result, pd.DataFrame([last_row])], ignore_index=True)
    return result[
        ['MD', 'INC', 'AZI', 'tangentX', 'tangentY', 'tangentZ', 'gamma', 'Radius', 'X', 'Y', 'Z', 'C_X', 'C_Y', 'C_Z']]


def visualise(df: pd.DataFrame, show_tangents=False, show_focuses=False, *args, **kwargs):
    """
    only for local test
    :param df:
    :param show_tangents:
    :param show_focuses:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Дуга
    ax.plot(df['X'], df['Y'], df['Z'], color='black', linewidth=2, label='Дуга')

    if show_focuses:
        ax.scatter(df['C_X'], df['C_Y'], df['C_Z'], color='blue', s=50, label='P_center')

    axis_length = 1250
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1)

    if show_tangents:
        ax.quiver(df['X'], df['Y'], df['Z'], 100 * df['tangentX'], 100 * df['tangentY'], 100 * df['tangentZ'],
                  color='blue', arrow_length_ratio=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-400, 400)
    ax.set_ylim(-400, 400)
    ax.set_zlim(-10, 2000)

    ax.legend()
    plt.show()
