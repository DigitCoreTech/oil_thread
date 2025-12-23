import numpy as np
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Координаты начальной точки
    N1 = 0
    E1 = 0
    TVD1 = 0
    print("N1 =", N1, ", E1 =", E1, ", TVD1 =", TVD1)

    # Первый замер
    MD1 = 0
    A1 = 0.0
    I1 = 0.0
    print("MD1 =", MD1, ", A1 =", A1, ", I1 =", I1)

    # Второй замер
    MD2 = 1000
    I2 = 50.0 * np.pi / 180.0
    A2 = 45.0 * np.pi / 180.0
    print("MD2 =", MD2, ", A2 =", A2, ", I2 =", I2)

    MD_delta = MD2 - MD1
    print("MD_delta = ", MD_delta)

    # Dogleg cos (cosθ)
    dogleg_cos = np.sin(I1) * np.sin(I2) * np.cos(A2 - A1) + np.cos(I1) * np.cos(I2)
    print("dogleg_cos =", dogleg_cos)
    # Dogleg Angle (θ)
    dogleg = np.arccos(dogleg_cos)
    print("dogleg =", dogleg)

    # Ratio Factor (RF)
    RF = np.tan(0.5 * dogleg) * 2.0 / dogleg
    print("RF =", RF)

    # Coordinate Changes
    delta_N = 0.5 * MD_delta * (np.sin(I1) * np.cos(A1) + np.sin(I2) * np.cos(A2)) * RF
    print("delta_N =", delta_N)
    delta_E = 0.5 * MD_delta * (np.sin(I1) * np.sin(A1) + np.sin(I2) * np.sin(A2)) * RF
    print("delta_E =", delta_E)
    delta_TVD = 0.5 * MD_delta * (np.cos(I1) + np.cos(I2)) * RF
    print("delta_TVD =", delta_TVD)

    # Accumulate Coordinates
    N2 = N1 + delta_N
    E2 = E1 + delta_E
    TVD2 = TVD1 + delta_TVD
    print("N2 =", N2, ", E2 =", E2, ", TVD2 =", TVD2)

    # Концы дуги
    P1 = np.array([N1, E1, TVD1])
    P2 = np.array([N2, E2, TVD2])
    print("P1 =", N1, ", P2 =", P2)

    # Calculate Dogleg Severity (DLS)
    DLS = (((dogleg * 180.0) / np.pi) / MD_delta) * 30
    print("DLS =", DLS, "degree / 30 ft")

    # Radius value
    R_dogleg = MD_delta / dogleg
    print("R_dogleg =", R_dogleg)
    R_dls = 1718.9 / DLS
    print("R_dls =", R_dls)
    radius = R_dogleg

    # Единичный вектор касательной в начальной точке
    t1 = np.array([np.sin(I1) * np.cos(A1), np.sin(I1) * np.sin(A1), np.cos(I1)])
    # Единичный вектор касательной в конечной точке
    t2 = np.array([np.sin(I2) * np.cos(A2), np.sin(I2) * np.sin(A2), np.cos(I2)])
    print("t1 =", t1, ", t2 =", t2)

    # Нормальный вектор к плоскости дуги
    t1_x_t2 = np.cross(t1, t2)
    normal = t1_x_t2 / np.linalg.norm(t1_x_t2)
    print("normal =", normal)

    # Вектор нормали кривизны (направление к центру)
    k = np.cross(normal, t1)

    # Координаты центра дуги
    N_center = N1 + R_dogleg * k[0]
    E_center = E1 + R_dogleg * k[1]
    TVD_center = TVD1 + R_dogleg * k[2]
    # print( "N_center =", N_center, "E_center =", E_center, "TVD_center =", TVD_center )
    P_center = np.array([N_center, E_center, TVD_center])
    print("P_center =", P_center)

    # Вектора от центра до концов дуги
    CP1 = P1 - P_center
    CP2 = P2 - P_center
    print("CP1 =", CP1, ", CP2 =", CP2)

    # Проверка (рекомендуется)
    mod_CP1 = np.linalg.norm(CP1)
    mod_CP2 = np.linalg.norm(CP2)
    print("mod_CP1 =", mod_CP1, ", mod_CP2 =", mod_CP2)

    # Ортонормированный базис в плоскости дуги
    e1 = CP1 / np.linalg.norm(CP1)
    normal = np.cross(CP1, CP2)
    e2 = np.cross(normal, e1)
    e2 = e2 / np.linalg.norm(e2)

    # Число отображаемых точек на дуге
    n_arc_points = 10

    # Параметризация угла
    t_values = np.linspace(0, dogleg, n_arc_points)
    print("t_values =", t_values)

    # Точки дуги
    arc = np.array([
        P_center + radius * (np.cos(tt) * e1 + np.sin(tt) * e2)
        for tt in t_values
    ])
    print("arc =", arc)

    # ===== ВИЗУАЛИЗАЦИЯ =====
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Дуга
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color='black', linewidth=2, label='Дуга')

    ax.scatter(*P1, color='red', s=50, label='P1')
    ax.scatter(*P2, color='green', s=50, label='P2')
    ax.scatter(*P_center, color='blue', s=50, label='P_center')

    axis_length = 1250
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-5, 1000)
    ax.set_ylim(-5, 1000)
    ax.set_zlim(0, 1000)

    ax.legend()
    plt.show()