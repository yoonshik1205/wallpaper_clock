#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime, timedelta, UTC
from PIL import Image, ImageDraw
import numpy as np
import json

from time import sleep

IMG_SIZE = (2560, 1600)

C_SIZE = (2560, 1600)
W, H = C_SIZE

CANV_R = 700

RING1_THICK = 40
RING2_THICK = 20
RING3_THICK = 20


C1X, C1Y = 380, 280
C1SCALE = 300 / (40 * 1.495978707e11)
C1RS = 10000

C2X, C2Y = 323, 740
C2SCALE = 200 / (2 * 1.495978707e11)
C2RS = 1000


BG_COLOR = (0, 0, 0)
# BG_COLOR = (255, 255, 255)
CLEAR_WHITE = (255, 255, 230, 100)
WHITE = (255, 255, 230)
FAINTWHITE = (180, 180, 160)
CLEAR_BLACK = (0, 0, 0, 80)
DARKGRAY = (60, 60, 50)
CLEAR_GRAY = (180, 180, 160, 100)

RED = (143, 48, 30)
YELLOW = (143, 119, 30)
DAYLIGHT = (48, 48, 40)

PLANETS = [
    [0, 0, 0, 0, 0, 0],
    [0.20563069, 0.38709893, 0.12214182, 48.33167, 77.45645, 252.25084],
    [0.00677323, 0.72333199, 0.05954315, 76.68069, 131.53298, 181.97973],
    [0.01671022, 1.00000011, 0, 348.73936, 102.94719, 100.46435],
    [0.09341233, 1.52366231, 0.03262693, 49.55954, 336.04084, 355.45332],
    [0.04839266, 5.20260319, 0.02258729, 100.47391, 14.72848, 34.39624],
    [0.0541506, 9.5549096, 0.04344155, 113.662424, 92.598878, 49.954244],
    [0.04716771, 19.21844606, 0.01337173, 74.016925, 170.954276, 313.23218],
    [0.00858587, 30.11038687, 0.03089337, 131.784057, 44.964762, 304.88003],
]
SIZES = [10, 3, 5, 0, 3, 4, 3, 2, 2]
COLORS = [
    WHITE,
    (179, 176, 171),
    (245, 235, 206),
    (54, 101, 158),
    (209, 82, 50),
    (237, 175, 104),
    (237, 211, 140),
    (188, 227, 217),
    (76, 152, 207),
]
TRUE_SIZES_KM = [
    0.,
    2439.4,
    6051.8,
    6371.0,
    3389.5,
    69911,
    58232,
    25362,
    24622
]
for i, p in enumerate(PLANETS):
    p[5] -= p[4]
    if p[5] < 0: p[5] += 360
    p[4] -= p[3]
    if p[4] < 0: p[4] += 360
    p.append(SIZES[i])
    p.append(COLORS[i])
    p.append('au')
    p.append(i==3)
    try: p.append(TRUE_SIZES_KM[i])
    except: p.append(0.)
    p.append(i)


def calc_planet_magnitude(idx:int, earth_xyz:tuple, xyz:tuple):
    if idx==0 or idx==3: return None
    ec = np.array(earth_xyz, np.float32) / 1.495978707e11
    pc = np.array(xyz, np.float32) / 1.495978707e11
    p2e = ec - pc
    d = np.sqrt((p2e**2).sum())
    r = np.sqrt((pc**2).sum())
    alpha = np.rad2deg(np.arccos(-(pc*p2e).sum()/d/r))
    alpha_poly = np.float_power(alpha, np.arange(7))
    first_term = 5. * np.log10(d*r)
    coefs = np.zeros(7)
    if idx==1:
        coefs = np.array([-0.613, 6.3280E-02, -1.6336E-03, 3.3644E-05, -3.4265E-07, 1.6893E-09, -3.0334E-12])
    elif idx==2 and alpha <= 163.7:
        coefs = np.array([-4.384, -1.044E-03, 3.687E-04, -2.814E-06, 8.938E-09, 0., 0.])
    elif idx==2:
        coefs = np.array([236.05828, -2.81914E+00, 8.39034E-03, 0., 0., 0., 0.])
    elif idx==4:
        coefs = np.array([-1.601, 0.02267, -0.0001302, 0., 0., 0., 0.])
    elif idx==5:
        coefs = np.array([-9.395, -3.7E-04, 6.16E-04, 0., 0., 0., 0.])
    elif idx==6:
        coefs = np.array([-8.95, -3.7E-04, 6.16E-04, 0., 0., 0., 0.])
    elif idx==7:
        coefs = np.array([-7.110, 6.587E-3, 1.045E-4, 0., 0., 0., 0.])
    elif idx==8:
        coefs = np.array([-7.00, 7.944E-3, 9.617E-5, 0., 0., 0., 0.])
    return first_term + (coefs * alpha_poly).sum()


LATLONG = (37.5, 127.0)
LAT, LONG = LATLONG
EQ_R = CANV_R * 90 / (180 - LAT)

GM = 1.32712440018e20
AX_TILT = 23.44 / 180 * np.pi
AX_TILT_MAT = np.array([[1, 0, 0],
                        [0, np.cos(AX_TILT), -np.sin(AX_TILT)],
                        [0, np.sin(AX_TILT), np.cos(AX_TILT)]]) # ecliptic coord to equatorial coord


class sky_obj(object):
    '''
    sky_coord: tuple (long, lat) in radians
    '''
    def __init__(self, r:float, fill=CLEAR_WHITE, sky_coord:tuple=(0., 0.)) -> None:
        self.sky_coord = sky_coord
        self.r, self.fill = r, fill
    @property
    def proj_coord(self):
        if not self.sky_coord: return None
        r = (np.pi/2-self.sky_coord[1]) * EQ_R * 2 / np.pi
        th = self.sky_coord[0]
        return (W/2 + r * np.cos(th), H/2 + r * np.sin(th)) if r < CANV_R else None
    def draw(self, draw:ImageDraw.ImageDraw) -> None:
        pc = self.proj_coord
        if pc: draw.ellipse((pc[0]-self.r, pc[1]-self.r, pc[0]+self.r, pc[1]+self.r), fill=self.fill)

class planet(sky_obj):
    earth = None
    def __init__(self, e, a, inc, asc, peri, M0, r:float, fill, unit="au", earth=False, true_r_km:float=0., idx:int=0) -> None:
        self.true_r_km = true_r_km
        self.r, self.fill = r, fill
        self.is_sun = False
        if a == 0:
            self.x, self.y, self.z = 0, 0, 0
            self.space_coord = (0, 0, 0)
            self.is_sun = True
            return
        if unit == "au": a *= 1.495978707e11
        elif unit == "km": a *= 1e3
        self.period = 2 * np.pi * np.power(a, 1.5) / np.sqrt(GM)
        self.e, self.a, self.inc, self.asc, self.peri, self.M0 = e, a, inc, asc, peri, M0
        self.x, self.y, self.z = self.get_space_coord(NOWUTC)
        self.space_coord = (self.x, self.y, self.z)
        if earth: planet.earth = self
        self.idx = idx
    def get_space_coord(self, tt:datetime):
        tp = tt - datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)
        M = 2 * np.pi * (tp.total_seconds() / self.period + self.M0 / 360)
        E0 = M
        E = M + self.e * np.sin(E0)
        while abs(E - E0) >= 1e-4:
            E0 = E
            E = M + self.e * np.sin(E)
        v = 2 * np.arctan(np.sqrt((1+self.e)/(1-self.e)) * np.tan(E/2))
        rr = self.a * (1 - self.e * np.cos(E))
        asc = self.asc / 180 * np.pi
        peri = self.peri / 180 * np.pi
        th = v + peri + asc
        zt = rr * np.sin(self.inc) * np.sin(v+peri)
        xt, yt = rr * np.cos(th), rr * np.sin(th) # approximation
        return AX_TILT_MAT @ np.array([xt, yt, zt])
    @classmethod
    def get_sky_coord_of_sun(cls, tt:datetime):
        x, y, z = cls.earth.get_space_coord(tt)
        return (np.arctan2(-y, -x), np.arctan2(-z, np.sqrt(x**2 + y**2)))
    @property
    def sky_coord(self):
        assert planet.earth
        if self is planet.earth: return None
        ex, ey, ez = planet.earth.space_coord
        fx, fy, fz = self.x - ex, self.y - ey, self.z - ez
        return (np.arctan2(fy, fx), np.arctan2(fz, np.sqrt(fx**2 + fy**2)))
    @property
    def earth_sqdist(self):
        assert planet.earth
        if self is planet.earth: return 0.
        ec = np.array(planet.earth.space_coord)
        sc = np.array(self.space_coord)
        return ((ec-sc)**2).sum()
    def draw(self, drw: ImageDraw.ImageDraw) -> None:
        if self is not planet.earth and self.true_r_km > 0:
            self.r = 0.5 + (4 - calc_planet_magnitude(self.idx, planet.earth.space_coord, self.space_coord)) / 2
            if self.r < 0: self.r = 0.
        super().draw(drw)
        if self.proj_coord:
            x, y = self.proj_coord
            r = 10
            drw.ellipse((x-r, y-r, x+r, y+r), outline=self.fill)
        x, y, _ = np.array(self.space_coord) @ AX_TILT_MAT
        if self.is_sun:
            drw.ellipse((C1X-10, C1Y-10, C1X+10, C1Y+10), self.fill)
            drw.ellipse((C2X-10, C2Y-10, C2X+10, C2Y+10), self.fill)
            return
        elif self.a > 3 * 1.495978707e11:
            dr = self.true_r_km / C1RS
            dx = x * C1SCALE + C1X
            dy = -y * C1SCALE + C1Y
        else:
            dr = self.true_r_km / C2RS
            dx = x * C2SCALE + C2X
            dy = -y * C2SCALE + C2Y
        drw.ellipse((dx-dr, dy-dr, dx+dr, dy+dr), self.fill)

class planet_orbit(object):
    def __init__(self, p:planet, thick:float, fill, subd:int=60) -> None:
        if p.is_sun: return
        self.thick, self.fill = thick, fill
        pp = p.period / subd
        dn = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)
        self.xy = []
        xyz = []
        for i in range(subd):
            x, y, z = p.get_space_coord(dn + timedelta(seconds=pp*i))
            xyz.append([x, y, z])
        xyz = np.array(xyz) @ AX_TILT_MAT
        if p.a > 3 * 1.495978707e11:
            for x, y, _ in xyz:
                self.xy.append((x * C1SCALE + C1X, -y * C1SCALE + C1Y))
        else:
            for x, y, _ in xyz:
                self.xy.append((x * C2SCALE + C2X, -y * C2SCALE + C2Y))
    def draw(self, draw:ImageDraw.ImageDraw):
        draw.polygon(self.xy, outline=self.fill, width=self.thick)

def space_to_proj(space):
    x, y, z = space
    long = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = (np.pi/2 - lat) * EQ_R * 2 / np.pi
    prox = W/2 + r * np.cos(long)
    proy = H/2 + r * np.sin(long)
    return (prox, proy)

def sky_to_space(sky):
    long, lat = sky
    z = np.sin(lat)
    x = np.cos(lat) * np.cos(long)
    y = np.cos(lat) * np.sin(long)
    return (x, y, z)

def space_to_sky(space):
    x, y, z = space
    long = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    return (long, lat)

def sky_to_proj(sky):
    long, lat = sky
    r = (np.pi/2 - lat) * EQ_R * 2 / np.pi
    return (W/2 + r * np.cos(long), H/2 + r * np.sin(long))

class approx_moon(object):
    '''
    approximately circular orbit because i don't want to deal with this
    '''
    def __init__(self, fill) -> None:
        ang_speed = 2*np.pi / timedelta(days=27.321662).total_seconds()
        node_ang_speed = - 2*np.pi / timedelta(days=6798.383).total_seconds()
        incl = 5.145 / 180 * np.pi
        node_0 = 23.8427 / 180 * np.pi
        node_at = datetime(2024, 6, 2, 18, 42, 58, tzinfo=UTC)
        long = node_0 + ang_speed * (NOWUTC - node_at).total_seconds()
        node = node_0 + node_ang_speed * (NOWUTC - node_at).total_seconds()
        occ = AX_TILT_MAT @ np.array(sky_to_space((node-np.pi/2, np.pi/2-incl)))
        ocx, ocy = space_to_sky(occ)
        self.orbitplane = great_circle(3, DARKGRAY, (ocx, ocy))
        v0 = np.array([np.cos(long-node), np.sin(long-node), 0])
        space = AX_TILT_MAT @ np.array([
            [np.cos(node), -np.sin(node), 0],
            [np.sin(node), np.cos(node), 0],
            [0, 0, 1]
        ]) @ np.array([
            [1, 0, 0],
            [0, np.cos(incl), -np.sin(incl)],
            [0, np.sin(incl), np.cos(incl)]
        ]) @ v0
        self.proj_coord = space_to_proj(space)
        self.fill = fill
    def draw(self, draw:ImageDraw.ImageDraw):
        r = SIZES[0]
        er = r/2
        px, py = self.proj_coord
        draw.ellipse((px-r, py-r, px+r, py+r), self.fill)
        draw.ellipse((px-er, py-er, px+er, py+er), BG_COLOR)

class great_circle(object):
    '''
    center: in sky coord (radians, longlat)
    '''
    def __init__(self, thick:float, fill, center:tuple=(0., 0.), subd:int=240):
        self.fill = fill
        self.thick = thick
        cx, cy = center
        mat = np.array([
            [np.cos(cx), -np.sin(cx), 0],
            [np.sin(cx), np.cos(cx), 0],
            [0, 0, 1]
        ]) @ np.array([
            [np.sin(cy), 0, np.cos(cy)],
            [0, 1, 0],
            [-np.cos(cy), 0, np.sin(cy)]
        ])
        vth = np.linspace(0, 2*np.pi, subd, False)
        vs = np.stack([np.cos(vth), np.sin(vth), np.zeros(subd)], 0)
        prox, proy = space_to_proj(mat @ vs)
        self.pts = list(zip(prox, proy))
    def draw(self, draw:ImageDraw.ImageDraw):
        # assume no great circle clips out of bounds
        draw.polygon(self.pts, outline=self.fill, width=self.thick)

class invisible_side_shade(object):
    '''
    the part that is under the horizon is tinted black
    '''
    def __init__(self, center:tuple, subd:int=240):
        cx, cy = center
        mat = np.array([
            [np.cos(cx), -np.sin(cx), 0],
            [np.sin(cx), np.cos(cx), 0],
            [0, 0, 1]
        ]) @ np.array([
            [np.sin(cy), 0, np.cos(cy)],
            [0, 1, 0],
            [-np.cos(cy), 0, np.sin(cy)]
        ])
        vth = np.linspace(0, 2*np.pi, subd, False)
        vs = np.stack([np.cos(vth), np.sin(vth), np.zeros(subd)], 0)
        prox, proy = space_to_proj(mat @ vs)
        prox2 = W/2 + CANV_R * np.cos(vth + cx)
        proy2 = H/2 + CANV_R * np.sin(vth + cx)
        self.pts = []
        self.pts.extend(zip(prox, proy))
        self.pts.extend(zip(prox2, proy2))
    def draw(self, draw:ImageDraw.ImageDraw):
        draw.polygon(self.pts, CLEAR_BLACK)

class half_great_circle(object):
    '''
    the top half instead of the bottom half
    '''
    def __init__(self, thick:float, fill, center:tuple=(0., 0.), subd:int=120):
        self.fill = fill
        self.thick = thick
        cx, cy = center
        mat = np.array([
            [np.cos(cx), -np.sin(cx), 0],
            [np.sin(cx), np.cos(cx), 0],
            [0, 0, 1]
        ]) @ np.array([
            [np.sin(cy), 0, np.cos(cy)],
            [0, 1, 0],
            [-np.cos(cy), 0, np.sin(cy)]
        ])
        vth = np.linspace(np.pi/2, np.pi*1.5, subd+1)
        vs = np.stack([np.cos(vth), np.sin(vth), np.zeros(subd+1)], 0)
        prox, proy = space_to_proj(mat @ vs)
        self.pts = list(zip(prox, proy))
    def draw(self, draw:ImageDraw.ImageDraw):
        draw.line(self.pts, fill=self.fill, width=self.thick)

class proj_circle(object):
    def __init__(self, thick:float, fill, r:float, outside:bool=True):
        self.fill = fill
        self.thick = thick
        self.r = r
        self.outside = outside
    def draw(self, draw:ImageDraw.ImageDraw):
        if not self.outside and self.r > CANV_R: return
        draw.ellipse((W/2-self.r, H/2-self.r, W/2+self.r, H/2+self.r), outline=self.fill, width=self.thick)

class proj_lines(object):
    def __init__(self, thick:float, fill, r:float, num:int, center:tuple=(W/2, H/2)):
        self.fill = fill
        self.thick = thick
        self.r = r
        self.num = num
        self.cx, self.cy = center
    def draw(self, draw:ImageDraw.ImageDraw):
        for th in np.linspace(0, np.pi, self.num//2, False):
            xy = [(self.cx + self.r * np.cos(th), self.cy + self.r * np.sin(th)),
                  (self.cx - self.r * np.cos(th), self.cy - self.r * np.sin(th))]
            draw.line(xy, fill=self.fill, width=self.thick)

def get_sunset_half_offset(sunphi:float):
    d = np.sin(sunphi) * np.tan(np.deg2rad(LAT))
    nr = np.cos(sunphi)
    mt = np.arcsin(d / nr)
    return np.pi/2 + mt

def get_sunset_south_angle(sunphi:float):
    dd = np.sin(sunphi) / np.cos(np.deg2rad(LAT))
    return np.pi/2 + np.arcsin(dd)

class daylight_polygon(object):
    def __init__(self, sunth:float, sunph:float, fill, subd:int=80):
        sho = get_sunset_half_offset(sunph)
        ssa = get_sunset_south_angle(sunph)
        self.fill = fill
        cy = LAT*np.pi/180
        cx = sunth + sho
        mat1 = np.array([
            [np.cos(cx), -np.sin(cx), 0],
            [np.sin(cx), np.cos(cx), 0],
            [0, 0, 1]
        ]) @ np.array([
            [np.sin(cy), 0, np.cos(cy)],
            [0, 1, 0],
            [-np.cos(cy), 0, np.sin(cy)]
        ])
        cx = sunth - sho
        mat2 = np.array([
            [np.cos(cx), -np.sin(cx), 0],
            [np.sin(cx), np.cos(cx), 0],
            [0, 0, 1]
        ]) @ np.array([
            [np.sin(cy), 0, np.cos(cy)],
            [0, 1, 0],
            [-np.cos(cy), 0, np.sin(cy)]
        ])
        vth1 = np.linspace(0, -ssa, subd, False)
        vs1 = np.stack([np.cos(vth1), np.sin(vth1), np.zeros(subd)], 0)
        prox1, proy1 = space_to_proj(mat1 @ vs1)
        verts = list(zip(prox1, proy1))
        vth2 = np.linspace(ssa, 0, subd, False)
        vs2 = np.stack([np.cos(vth2), np.sin(vth2), np.zeros(subd)], 0)
        prox2, proy2 = space_to_proj(mat2 @ vs2)
        verts.extend(zip(prox2, proy2))
        vth3 = np.linspace(sunth-sho, sunth+sho, subd*2, False)
        prox3 = W/2 + CANV_R * np.cos(vth3)
        proy3 = H/2 + CANV_R * np.sin(vth3)
        verts.extend(zip(prox3, proy3))
        self.verts = verts
    def draw(self, draw:ImageDraw.ImageDraw):
        draw.polygon(self.verts, self.fill)

def point_out_of_canv(x, y):
    return (W/2-x)**2+(H/2-y)**2>CANV_R**2

def get_intersect_with_canv(x1, y1, x2, y2):
    '''
    x1, y1 is inside, x2, y2 is outside
    '''
    dx, dy = x2-x1, y2-y1
    nm = np.sqrt(dx**2 + dy**2)
    dx /= nm
    dy /= nm
    ds = dx*(W/2 - x1) + dy*(H/2 - y1)
    t = ds + np.sqrt(ds**2 - (W/2 - x1)**2 - (H/2 - y1)**2 + CANV_R**2)
    rx = x1 + t * dx
    ry = y1 + t * dy
    return (rx, ry)

class generic_line(object):
    def __init__(self, xy, thick:float, fill, clip_at_canv:bool=False):
        self.xy = xy
        self.thick = thick
        self.fill = fill
        self.valid = True
        if clip_at_canv:
            if len(xy)==2: (x1, y1), (x2, y2) = xy
            else: x1, y1, x2, y2 = xy
            if point_out_of_canv(x1, y1):
                if point_out_of_canv(x2, y2): self.valid = False
                else:
                    rx, ry = get_intersect_with_canv(x2, y2, x1, y1)
                    self.xy = (x2, y2, rx, ry)
            elif point_out_of_canv(x2, y2):
                rx, ry = get_intersect_with_canv(x1, y1, x2, y2)
                self.xy = (x1, y1, rx, ry)
    def draw(self, draw:ImageDraw.ImageDraw):
        if self.valid: draw.line(self.xy, self.fill, self.thick)

class generic_dot(object):
    def __init__(self, pos:tuple, r:float, fill):
        x, y = pos
        self.xy = (x-r, y-r, x+r, y+r)
        self.fill = fill
    def draw(self, draw:ImageDraw.ImageDraw):
        draw.ellipse(self.xy, self.fill)

class generic_circle(object):
    def __init__(self, pos:tuple, r:float, outline, thick:float, fill=None):
        x, y = pos
        self.xy = (x-r, y-r, x+r, y+r)
        self.fill = fill
        self.thick = thick
        self.outline = outline
    def draw(self, draw:ImageDraw.ImageDraw):
        draw.ellipse(self.xy, self.fill, self.outline, self.thick)

def calc_24_xys():
    ith = np.linspace(-np.pi/2, 1.5*np.pi, 24, False)
    ivs = np.stack([np.cos(ith), np.sin(ith), np.zeros_like(ith)], 0)
    rx, ry, _ = AX_TILT_MAT @ ivs
    angs = np.arctan2(ry, rx)
    p1 = zip(W/2 + CANV_R*np.cos(angs), H/2 + CANV_R*np.sin(angs))
    p2 = zip(W/2 + (CANV_R+RING1_THICK-1)*np.cos(angs), H/2 + (CANV_R+RING1_THICK-1)*np.sin(angs))
    thick = []
    for i in range(24):
        if not i%6: thick.append(5)
        elif not i%3: thick.append(3)
        else: thick.append(1)
    return zip(zip(p1, p2), thick)

def calc_month_dates_xys():
    ny = NOWLOCAL.year
    dates = []
    thick = []
    for mth in range(1, 13):
        for dy in range(1, 32):
            try:
                dates.append(datetime(ny, mth, dy, 0, 0, 0).astimezone(UTC))
                if mth==1 and dy==1: thick.append(6)
                elif dy==1: thick.append(4)
                elif dy%10==1: thick.append(2)
                else: thick.append(1)
            except ValueError: continue
    angs = []
    for dt in dates:
        long, _ = planet.get_sky_coord_of_sun(dt)
        angs.append(long)
    angs = np.array(angs)
    p1 = zip(W/2 + (CANV_R+RING1_THICK)*np.cos(angs), H/2 + (CANV_R+RING1_THICK)*np.sin(angs))
    p2 = zip(W/2 + (CANV_R+RING1_THICK+RING2_THICK-1)*np.cos(angs), H/2 + (CANV_R+RING1_THICK+RING2_THICK-1)*np.sin(angs))
    return zip(zip(p1, p2), thick)

def calc_time_xys(suntheta:float):
    nu = NOWUTC
    true_midnight = datetime(nu.year, nu.month, nu.day, 0, 0, 0, tzinfo=UTC) - timedelta(hours=LONG/15)
    false_midnight = datetime(nu.year, nu.month, nu.day, 0, 0, 0).astimezone(UTC)
    midnight_theta = suntheta + (false_midnight - true_midnight).total_seconds() * 2*np.pi / 86400
    ith = np.linspace(midnight_theta, midnight_theta+2*np.pi, 24*6, False)
    thicks = np.ones(24*6, dtype=int) * 3
    thicks[::18] = 5
    lens = np.ones(24*6, dtype=int) * 10
    lens[::6] = 20
    lens[::18] = 30
    lens[::72] = 50
    p1 = zip(W/2 + (CANV_R-1)*np.cos(ith), H/2 + (CANV_R-1)*np.sin(ith))
    p2 = zip(W/2 + (CANV_R-lens)*np.cos(ith), H/2 + (CANV_R-lens)*np.sin(ith))
    return zip(zip(p1, p2), thicks)

class orbit_grids(object):
    def __init__(self):
        th = [generic_circle((C2X, C2Y), 180, WHITE, 3)]
        # for r in [50, 100, 150]: th.append(generic_circle((C2X, C2Y), r, CLEAR_WHITE, 1))
        th.append(proj_lines(1, CLEAR_WHITE, 180, 24, (C2X, C2Y)))
        th.append(generic_circle((C1X, C1Y), 260, WHITE, 3))
        # for r in [50, 100, 150, 200, 250]: th.append(generic_circle((C1X, C1Y), r, CLEAR_WHITE, 1))
        th.append(proj_lines(1, CLEAR_WHITE, 260, 24, (C1X, C1Y)))
        self.things = th
    def draw(self, draw:ImageDraw.ImageDraw):
        for t in self.things:
            t.draw(draw)

class skywindow(object):
    def __init__(self) -> None:
        planets = [planet(*p) for p in PLANETS[::-1]]
        suntheta, sunphi = planets[-1].sky_coord

        ps = sorted(planets, key=lambda x: x.earth_sqdist, reverse=True)

        nu = NOWUTC
        true_midnight = datetime(nu.year, nu.month, nu.day, 0, 0, 0, tzinfo=UTC) - timedelta(hours=LONG/15)
        daytheta = (nu - true_midnight).total_seconds() * 2 * np.pi / 86400
        hand_dir = daytheta + suntheta + np.pi
        # hand_marker_xy = (W/2 - (EQ_R*2-CANV_R)*np.cos(hand_dir), H/2 - (EQ_R*2-CANV_R)*np.sin(hand_dir),
        #                     W/2 - (H/2-5)*np.cos(hand_dir), H/2 - (H/2-5)*np.sin(hand_dir))
        hand_marker_xy = (W/2 - (EQ_R*2-CANV_R)*np.cos(hand_dir), H/2 - (EQ_R*2-CANV_R)*np.sin(hand_dir),
                            W/2 - (CANV_R-7)*np.cos(hand_dir), H/2 - (CANV_R-7)*np.sin(hand_dir))
        other_xy = (W/2 - (EQ_R*2-CANV_R)*np.cos(hand_dir), H/2 - (EQ_R*2-CANV_R)*np.sin(hand_dir),
                    W/2 + CANV_R*np.cos(hand_dir), H/2 + CANV_R*np.sin(hand_dir))

        sunhand_xy = (W/2, H/2, W/2 + (H/2-5)*np.cos(suntheta), H/2 + (H/2-5)*np.sin(suntheta))

        moon = approx_moon(FAINTWHITE)
        moonline_xy = (W/2, H/2) + moon.proj_coord

        self.things = []

        self.things.append(orbit_grids())
        self.things.extend(planet_orbits)
        self.things.append(daylight_polygon(suntheta, sunphi, DAYLIGHT))
        self.things.extend([proj_circle(1, CLEAR_WHITE, r, False) for r in (np.arange(1, 12)*EQ_R/6)]) # latitude
        self.things.extend([
            proj_circle(1, WHITE, EQ_R*2-CANV_R), # circle around pole
            proj_lines(1, CLEAR_WHITE, CANV_R, 24), # longitude
            moon.orbitplane,
            proj_circle(3, RED, EQ_R), # equator
            great_circle(3, YELLOW, (-np.pi/2, np.pi/2-AX_TILT)), # ecliptic
        ])
        self.things.extend(lines)
        self.things.extend(stars)
        self.things.extend([
            generic_line(sunhand_xy, 3, WHITE), # line from pole past sun
            generic_dot(sunhand_xy[2:], 5, WHITE), # dot on sun hand
        ])
        self.things.extend(ps)
        self.things.extend([
            generic_line(moonline_xy, 3, FAINTWHITE), # line from pole to moon
            moon,
            invisible_side_shade((hand_dir, LAT*np.pi/180)),
            great_circle(3, FAINTWHITE, (hand_dir, LAT*np.pi/180)), # horizon
            half_great_circle(1, FAINTWHITE, (hand_dir+np.pi, (90-LAT)*np.pi/180)), # east-west line
            generic_line(other_xy, 1, FAINTWHITE), # north-south line
        ])
        self.things.extend([generic_line(xy, th, WHITE) for xy, th in calc_time_xys(suntheta)]) # time
        self.things.extend([generic_line(xy, th, WHITE) for xy, th in calc_24_xys()]) # 24 seasons
        self.things.append(proj_circle(3, WHITE, CANV_R+RING1_THICK))
        self.things.extend([generic_line(xy, th, WHITE) for xy, th in calc_month_dates_xys()]) # midnight each day
        self.things.append(proj_circle(3, WHITE, CANV_R+RING1_THICK+RING2_THICK))
        self.things.extend([
            proj_circle(3, WHITE, CANV_R), # outline of sky
            generic_dot((W/2, H/2), 5, WHITE), # dot at center
            generic_line(hand_marker_xy, 3, FAINTWHITE), # time hand line
            generic_dot(hand_marker_xy[2:], 7, FAINTWHITE),
            generic_dot(hand_marker_xy[2:], 2, BG_COLOR), # time hand dot
        ])
    def draw(self, draw:ImageDraw.ImageDraw):
        for t in self.things:
            t.draw(draw)

if __name__=="__main__":

    print('clock started', flush=True)

    savepath = 'test.png'

    stard = 'stardata.txt'
    lined = 'linedata.txt'

    with open(stard) as j:
        parms = json.load(j)
    stars = [sky_obj(r, FAINTWHITE, (long, lat)) for r, long, lat in parms]
    with open(lined) as j:
        pines = json.load(j)
    lines = [generic_line(pine, 1, CLEAR_GRAY, True) for pine in pines]

    NOWUTC = datetime.now(tz=UTC)
    NOWLOCAL = datetime.now()
    planet_orbits = [planet_orbit(planet(*p), 3, DARKGRAY) for p in PLANETS[1:]]

    img = Image.new('RGB', C_SIZE, BG_COLOR)

    while True:
        NOWUTC = datetime.now(tz=UTC) + timedelta(minutes=1)
        NOWLOCAL = datetime.now() + timedelta(minutes=1)

        img1 = img.copy()
        idr = ImageDraw.Draw(img1, 'RGBA')
        sw = skywindow()
        sw.draw(idr)
        img1.save(savepath)

        # break

        sleep(120)

