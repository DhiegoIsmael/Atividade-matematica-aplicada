import math
import sys
import numpy as np
import pygame

# ============================================================
# Platform Hunter - Álgebra Linear aplicada a Jogos (versão melhorada)
#
# Dependências: pip install pygame numpy
#
# Controles:
#   A/D ou setas: mover | W/Espaço: pular | SHIFT: dash
#   Mouse esquerdo: atirar | TAB: mostrar vetores | R: reiniciar
#
# Novidades (marcadas com "NOVO"):
#   cone de visão, dash, tiro, ricochete, cristais orbitando,
#   deformação do jogador, câmera suave e minimapa.
# ============================================================

pygame.init()

WIDTH, HEIGHT = 1000, 600
FPS = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Álgebra Linear - Platform Hunter (versão melhorada)")
clock = pygame.time.Clock()

small_font = pygame.font.Font(None, 22)
title_font = pygame.font.Font(None, 36)

vision_overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

WORLD_WIDTH = 2400
WORLD_HEIGHT = 900
GRAVITY = 1200.0

# Base canônica de R^2
E1 = np.array([1.0, 0.0])
E2 = np.array([0.0, 1.0])

# Plataformas: x, y, largura, altura
platforms = [
    pygame.Rect(0, 540, 700, 60),
    pygame.Rect(780, 470, 260, 40),
    pygame.Rect(1120, 390, 260, 40),
    pygame.Rect(1450, 500, 300, 40),
    pygame.Rect(1780, 410, 280, 40),
    pygame.Rect(2120, 520, 280, 60),
    pygame.Rect(900, 650, 300, 35),
    pygame.Rect(1320, 700, 280, 35),
]

goal_rect = pygame.Rect(2320, 440, 30, 80)

BG = (25, 30, 42)
PLATFORM_COLOR = (85, 105, 130)
PLAYER_COLOR = (70, 180, 255)
ENEMY_COLOR = (235, 80, 90)
TEXT_COLOR = (235, 240, 245)
VISION_COLOR = (240, 190, 70)
GOAL_COLOR = (90, 220, 120)
CRYSTAL_COLOR = (190, 130, 255)
PROJECTILE_COLOR = (255, 255, 255)

# ------------------------------------------------------------
# Funções de Álgebra Linear
# ------------------------------------------------------------

def normalize(v):
    """v / ||v||"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.copy()
    return v / norm


def rotation_matrix(theta):
    """Matriz de rotação 2D."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c, -s],
        [s,  c]
    ], dtype=float)


def transform_points(points, position, angle, scale=1.0):
    """p_mundo = R S p_local + posição"""
    S = np.array([
        [scale, 0.0],
        [0.0, scale]
    ])

    R = rotation_matrix(angle)
    A = R @ S

    result = []
    for point in points:
        p = np.array(point, dtype=float)
        world = A @ p + position
        result.append(tuple(world))
    return result


def scale_matrix(sx, sy):
    """NOVO: matriz de escala diagonal."""
    return np.array([
        [sx, 0.0],
        [0.0, sy]
    ], dtype=float)


def shear_matrix(kx, ky=0.0):
    """NOVO: matriz de cisalhamento (inclina a figura)."""
    return np.array([
        [1.0, kx],
        [ky, 1.0]
    ], dtype=float)


def reflection_matrix(n):
    """NOVO: reflexão H = I - 2nnᵀ (usada no ricochete)."""
    n = normalize(np.array(n, dtype=float))
    return np.eye(2) - 2.0 * np.outer(n, n)


def apply_matrix(points, A, position):
    """NOVO: aplica uma matriz qualquer aos pontos locais."""
    return [
        tuple(A @ np.array(p, dtype=float) + position)
        for p in points
    ]


def draw_arrow(surface, color, start, vec, width=2):
    """NOVO: desenha um vetor como seta."""
    start = np.array(start, dtype=float)
    vec = np.array(vec, dtype=float)
    end = start + vec

    pygame.draw.line(surface, color, tuple(start), tuple(end), width)

    length = np.linalg.norm(vec)
    if length < 6:
        return

    # A ponta da seta usa rotações de ±25°
    back = -vec / length * 10.0
    head1 = end + rotation_matrix(math.radians(25)) @ back
    head2 = end + rotation_matrix(math.radians(-25)) @ back
    pygame.draw.polygon(surface, color, [tuple(end), tuple(head1), tuple(head2)])


# ------------------------------------------------------------
# Jogador
# ------------------------------------------------------------

class Player:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = np.array([120.0, 450.0], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.width = 36
        self.height = 50
        self.speed = 260.0
        self.jump_speed = -600.0
        self.on_ground = False

        # NOVO: dash, tiro e forma deformável
        self.facing = 1.0
        self.dash_speed = 750.0
        self.dash_time = 0.0
        self.dash_cooldown = 0.0
        self.dash_dir = np.array([1.0, 0.0])
        self.shoot_cooldown = 0.0
        self.shape_matrix = np.eye(2)

    @property
    def rect(self):
        return pygame.Rect(
            int(self.pos[0]),
            int(self.pos[1]),
            self.width,
            self.height
        )

    @property
    def center(self):
        return self.pos + np.array([
            self.width / 2,
            self.height / 2
        ])

    def update(self, dt, keys):
        move = 0.0

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move += 1.0

        if move != 0.0:
            self.facing = move

        up_pressed = (keys[pygame.K_w] or keys[pygame.K_SPACE]
                      or keys[pygame.K_UP])

        self.dash_cooldown = max(0.0, self.dash_cooldown - dt)
        self.dash_time = max(0.0, self.dash_time - dt)
        self.shoot_cooldown = max(0.0, self.shoot_cooldown - dt)

        # NOVO: dash = vetor de intenção normalizado * escalar
        wants_dash = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        if wants_dash and self.dash_cooldown <= 0.0 and self.dash_time <= 0.0:
            intent = np.array([move, -1.0 if up_pressed else 0.0])
            if np.linalg.norm(intent) == 0.0:
                intent = np.array([self.facing, 0.0])
            self.dash_dir = normalize(intent)
            self.dash_time = 0.15
            self.dash_cooldown = 0.9

        if self.dash_time > 0.0:
            self.velocity = self.dash_speed * self.dash_dir
        else:
            self.velocity[0] = move * self.speed

            if up_pressed and self.on_ground:
                self.velocity[1] = self.jump_speed
                self.on_ground = False

            # Gravidade como vetor
            gravity = np.array([0.0, GRAVITY])
            self.velocity += gravity * dt

        # p(t + dt) = p(t) + v * dt
        self.pos += self.velocity * dt

        self._resolve_collisions()

        self.pos[0] = max(0, min(self.pos[0], WORLD_WIDTH - self.width))

        if self.pos[1] > WORLD_HEIGHT:
            self.reset()

        self._update_shape()

    def _update_shape(self):
        # NOVO: M = H S. A escala S = diag(1/k, k) tem det = 1 (área
        # constante) e o cisalhamento H inclina o corpo ao correr.
        k = 1.0 + min(abs(self.velocity[1]) / 1500.0, 0.35)
        S = scale_matrix(1.0 / k, k)
        H = shear_matrix(-self.velocity[0] / 1800.0)
        self.shape_matrix = H @ S

    def _resolve_collisions(self):
        rect = self.rect
        self.on_ground = False

        for platform in platforms:
            if rect.colliderect(platform):
                previous_bottom = self.pos[1] + self.height - self.velocity[1] / FPS

                # Pousando sobre a plataforma
                if self.velocity[1] >= 0 and previous_bottom <= platform.top + 8:
                    self.pos[1] = platform.top - self.height
                    self.velocity[1] = 0
                    self.on_ground = True
                    rect = self.rect

                # Batendo na parte inferior
                elif self.velocity[1] < 0 and self.pos[1] >= platform.bottom - 5:
                    self.pos[1] = platform.bottom
                    self.velocity[1] = 0
                    rect = self.rect


# ------------------------------------------------------------
# NOVO: Projétil
# ------------------------------------------------------------

class Projectile:
    SPEED = 650.0

    def __init__(self, origin, target):
        self.pos = np.array(origin, dtype=float)
        # Direção unitária (alvo - origem) / ||alvo - origem||
        self.vel = self.SPEED * normalize(np.array(target, dtype=float) - self.pos)
        self.life = 2.0
        self.bounces = 0
        self.alive = True

    def update(self, dt):
        previous = self.pos.copy()
        self.pos = self.pos + self.vel * dt
        self.life -= dt

        for platform in platforms:
            if platform.collidepoint(self.pos[0], self.pos[1]):
                # Normal da superfície atingida
                if platform.left <= previous[0] <= platform.right:
                    ny = -1.0 if previous[1] < platform.top else 1.0
                    normal = np.array([0.0, ny])
                else:
                    nx = -1.0 if previous[0] < platform.left else 1.0
                    normal = np.array([nx, 0.0])

                # Ricochete: v' = H v
                self.pos = previous
                self.vel = reflection_matrix(normal) @ self.vel
                self.bounces += 1
                break

        out_of_world = (
            self.pos[0] < 0 or self.pos[0] > WORLD_WIDTH
            or self.pos[1] < -200 or self.pos[1] > WORLD_HEIGHT
        )

        if self.life <= 0.0 or self.bounces > 2 or out_of_world:
            self.alive = False


# ------------------------------------------------------------
# NOVO: Cristal que orbita um ponto
# ------------------------------------------------------------

class Crystal:
    def __init__(self, center, radius, omega, phase):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.omega = omega
        self.phase = phase
        self.collected = False

    def position(self, t):
        # p(t) = c + R(ωt + φ) (r, 0)
        angle = self.omega * t + self.phase
        return self.center + rotation_matrix(angle) @ np.array([self.radius, 0.0])


def make_crystals():
    return [
        Crystal((910, 395), 55, 1.6, 0.0),
        Crystal((1250, 315), 55, -1.4, 1.0),
        Crystal((1600, 425), 60, 1.8, 2.0),
        Crystal((1920, 335), 60, -1.5, 3.0),
        Crystal((2260, 445), 50, 1.7, 4.0),
        Crystal((1050, 605), 40, 2.0, 5.0),
    ]


# ------------------------------------------------------------
# Inimigo
# ------------------------------------------------------------

class Enemy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = np.array([560.0, 455.0], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.max_speed = 115.0
        self.vision_distance = 420.0

        # Campo de visão de 80 graus
        self.fov_half_angle = math.radians(40.0)
        self.fov_threshold = math.cos(self.fov_half_angle)

        self.forward = np.array([-1.0, 0.0], dtype=float)
        self.angle = math.pi

        # NOVO: vida, empurrão e renascimento
        self.hp = 3
        self.alive = True
        self.respawn_timer = 0.0
        self.knock = np.array([0.0, 0.0], dtype=float)

    @property
    def rect(self):
        return pygame.Rect(
            int(self.pos[0] - 20),
            int(self.pos[1] - 20),
            40,
            40
        )

    def hit(self, projectile_velocity):
        # NOVO: o empurrão é um múltiplo escalar da velocidade do projétil
        self.hp -= 1
        self.knock = self.knock + 0.5 * projectile_velocity
        if self.hp <= 0:
            self.alive = False
            self.respawn_timer = 4.0
            return True
        return False

    def update(self, player, dt):
        # Vetor do inimigo até o jogador
        direction = player.center - self.pos
        distance = np.linalg.norm(direction)

        if distance > 0:
            direction_unit = direction / distance
        else:
            direction_unit = np.zeros(2)

        if not self.alive:
            self.respawn_timer -= dt
            if self.respawn_timer <= 0.0:
                self.reset()
            return distance, 0.0, False

        # Produto interno = cos do ângulo entre forward e o alvo
        dot = float(np.dot(self.forward, direction_unit))

        can_see_player = (
            distance <= self.vision_distance
            and dot >= self.fov_threshold
        )

        if can_see_player:
            self.velocity = self.max_speed * direction_unit
            self.angle = math.atan2(direction_unit[1], direction_unit[0])
            self.forward = rotation_matrix(self.angle) @ E1
        else:
            # Patrulha para a esquerda
            self.velocity = np.array([-50.0, 0.0])
            self.angle = math.pi
            self.forward = -E1

        # NOVO: soma de vetores (velocidade própria + empurrão)
        self.pos += (self.velocity + self.knock) * dt
        self.knock = self.knock * math.exp(-6.0 * dt)

        self.pos[0] = max(120, min(self.pos[0], 900))

        if self.rect.colliderect(player.rect):
            player.reset()
            self.reset()

        return distance, dot, can_see_player


# ------------------------------------------------------------
# Desenho
# ------------------------------------------------------------

def draw_enemy(surface, enemy, camera_x):
    # Triângulo local transformado por R(θ) p + posição
    local_points = [(28, 0), (-18, -16), (-18, 16)]

    world_points = transform_points(
        local_points,
        enemy.pos,
        enemy.angle,
        scale=1.0
    )

    screen_points = [
        (int(x - camera_x), int(y))
        for x, y in world_points
    ]

    pygame.draw.polygon(surface, ENEMY_COLOR, screen_points)

    bar_x = int(enemy.pos[0] - camera_x) - 20
    bar_y = int(enemy.pos[1]) - 34
    pygame.draw.rect(surface, (60, 30, 35), (bar_x, bar_y, 40, 5))
    pygame.draw.rect(surface, ENEMY_COLOR, (bar_x, bar_y, int(40 * enemy.hp / 3), 5))


def draw_player(surface, player, camera_x):
    # NOVO: retângulo local transformado pela matriz de forma M = H S
    w, h = player.width / 2, player.height / 2
    local_points = [(-w, -h), (w, -h), (w, h), (-w, h)]

    center = player.center - np.array([camera_x, 0.0])
    points = apply_matrix(local_points, player.shape_matrix, center)
    pygame.draw.polygon(surface, PLAYER_COLOR, [(int(x), int(y)) for x, y in points])

    eye = apply_matrix([(player.facing * 8, -10)], player.shape_matrix, center)[0]
    pygame.draw.circle(surface, (15, 25, 40), (int(eye[0]), int(eye[1])), 4)


def draw_platforms(surface, camera_x):
    for platform in platforms:
        rect = platform.copy()
        rect.x -= int(camera_x)
        pygame.draw.rect(surface, PLATFORM_COLOR, rect, border_radius=4)


def draw_goal(surface, camera_x):
    rect = goal_rect.copy()
    rect.x -= int(camera_x)

    pygame.draw.rect(surface, GOAL_COLOR, rect)

    pygame.draw.line(
        surface, GOAL_COLOR,
        (rect.left, rect.top), (rect.left, rect.top - 60), 4
    )

    pygame.draw.polygon(
        surface,
        GOAL_COLOR,
        [
            (rect.left + 2, rect.top - 58),
            (rect.left + 35, rect.top - 48),
            (rect.left + 2, rect.top - 38),
        ]
    )


def draw_crystals(surface, crystals, camera_x, t):
    diamond = [(0, -12), (9, 0), (0, 12), (-9, 0)]

    for c in crystals:
        if c.collected:
            continue

        orbit_center = (int(c.center[0] - camera_x), int(c.center[1]))
        pygame.draw.circle(surface, (60, 55, 85), orbit_center, int(c.radius), 1)

        # O cristal orbita e também gira em torno de si
        pos = c.position(t)
        pts = transform_points(diamond, pos, 2.5 * t + c.phase)
        pygame.draw.polygon(
            surface, CRYSTAL_COLOR,
            [(int(x - camera_x), int(y)) for x, y in pts]
        )


def draw_projectiles(surface, projectiles, camera_x):
    for p in projectiles:
        pos = (int(p.pos[0] - camera_x), int(p.pos[1]))
        pygame.draw.circle(surface, PROJECTILE_COLOR, pos, 5)
        tail = p.pos - normalize(p.vel) * 16
        pygame.draw.line(
            surface, (160, 170, 190),
            pos, (int(tail[0] - camera_x), int(tail[1])), 2
        )


def draw_vision(surface, enemy, camera_x, distance, dot, can_see):
    # NOVO: cone real de visão. Cada ponto do arco é R(θ) forward * d
    vision_overlay.fill((0, 0, 0, 0))

    center = np.array([enemy.pos[0] - camera_x, enemy.pos[1]])
    steps = 16
    points = [tuple(center)]

    for k in range(steps + 1):
        theta = -enemy.fov_half_angle + 2.0 * enemy.fov_half_angle * k / steps
        direction = rotation_matrix(theta) @ enemy.forward
        points.append(tuple(center + direction * enemy.vision_distance))

    fill = (235, 80, 90, 55) if can_see else (240, 190, 70, 35)
    pygame.draw.polygon(vision_overlay, fill, points)

    for sign in (-1, 1):
        edge = rotation_matrix(sign * enemy.fov_half_angle) @ enemy.forward
        end = center + edge * enemy.vision_distance
        pygame.draw.line(
            vision_overlay, (240, 190, 70, 160), tuple(center), tuple(end), 1
        )

    surface.blit(vision_overlay, (0, 0))
    draw_arrow(surface, VISION_COLOR, center, enemy.forward * 60, 2)


def draw_debug_vectors(surface, player, enemy, camera_x, aim):
    # NOVO: v = (v·e1) e1 + (v·e2) e2
    c = player.center - np.array([camera_x, 0.0])
    v = player.velocity

    draw_arrow(surface, (255, 255, 255), c, v * 0.25, 2)
    draw_arrow(surface, (255, 110, 110), c, float(np.dot(v, E1)) * E1 * 0.25, 2)
    draw_arrow(surface, (110, 255, 150), c, float(np.dot(v, E2)) * E2 * 0.25, 2)

    # Vetor unitário de mira
    draw_arrow(surface, CRYSTAL_COLOR, c, aim * 55, 2)

    if enemy.alive:
        e = np.array([enemy.pos[0] - camera_x, enemy.pos[1]])
        pygame.draw.line(surface, (90, 100, 120), tuple(c), tuple(e), 1)


def draw_minimap(surface, player, enemy, crystals, camera_x, t):
    # NOVO: transformação afim p_mini = M p + b, com M = diag(sx, sy)
    MW, MH = 220, 80
    b = np.array([WIDTH - MW - 16.0, 16.0])
    M = scale_matrix(MW / WORLD_WIDTH, MH / WORLD_HEIGHT)

    def to_mini(p):
        return M @ np.array(p, dtype=float) + b

    pygame.draw.rect(surface, (15, 18, 28), (b[0], b[1], MW, MH))
    pygame.draw.rect(surface, (70, 80, 100), (b[0], b[1], MW, MH), 1)

    for p in platforms:
        tl = to_mini((p.x, p.y))
        size = M @ np.array([p.w, p.h], dtype=float)
        pygame.draw.rect(
            surface, PLATFORM_COLOR,
            (tl[0], tl[1], max(size[0], 1), max(size[1], 2))
        )

    goal = to_mini(goal_rect.center)
    pygame.draw.circle(surface, GOAL_COLOR, (int(goal[0]), int(goal[1])), 3)

    for c in crystals:
        if not c.collected:
            q = to_mini(c.position(t))
            pygame.draw.circle(surface, CRYSTAL_COLOR, (int(q[0]), int(q[1])), 2)

    if enemy.alive:
        q = to_mini(enemy.pos)
        pygame.draw.circle(surface, ENEMY_COLOR, (int(q[0]), int(q[1])), 3)

    q = to_mini(player.center)
    pygame.draw.circle(surface, PLAYER_COLOR, (int(q[0]), int(q[1])), 3)

    cam_tl = to_mini((camera_x, 0))
    cam_size = M @ np.array([WIDTH, HEIGHT], dtype=float)
    pygame.draw.rect(
        surface, (255, 255, 255),
        (cam_tl[0], cam_tl[1], cam_size[0], cam_size[1]), 1
    )


def draw_hud(surface, player, enemy, distance, dot, can_see, score, aim):
    title = title_font.render(
        "Platform Hunter - Álgebra Linear",
        True,
        TEXT_COLOR
    )
    surface.blit(title, (18, 16))

    det = float(np.linalg.det(player.shape_matrix))

    lines = [
        f"Player position = ({player.pos[0]:.1f}, {player.pos[1]:.1f})",
        f"Player velocity = ({player.velocity[0]:.1f}, {player.velocity[1]:.1f})",
        f"Enemy position  = ({enemy.pos[0]:.1f}, {enemy.pos[1]:.1f})",
        f"distance = {distance:.2f}",
        f"forward . direction_unit = {dot:.3f}",
        f"field of view = {'SIM' if can_see else 'NAO'}",
        f"aim = (m - p)/||m - p|| = ({aim[0]:.2f}, {aim[1]:.2f})",
        f"det(M_jogador) = {det:.3f}",
        f"dash cooldown = {player.dash_cooldown:.2f}s   score = {score}",
        "A/D: mover | W/Espaco: pular | SHIFT: dash | Mouse: atirar",
        "TAB: vetores | R: reiniciar",
    ]

    y = 58
    for line in lines:
        text = small_font.render(line, True, TEXT_COLOR)
        surface.blit(text, (18, y))
        y += 23


def draw_background(surface, camera_x):
    surface.fill(BG)

    grid_size = 50
    start_x = -(int(camera_x) % grid_size)

    for x in range(start_x, WIDTH, grid_size):
        pygame.draw.line(surface, (38, 44, 58), (x, 0), (x, HEIGHT), 1)

    for y in range(0, HEIGHT, grid_size):
        pygame.draw.line(surface, (38, 44, 58), (0, y), (WIDTH, y), 1)


# ------------------------------------------------------------
# Loop principal
# ------------------------------------------------------------

def main(max_frames=None):
    player = Player()
    enemy = Enemy()
    crystals = make_crystals()
    projectiles = []

    score = 0
    t = 0.0
    camera_x = 0.0
    show_debug = True
    frame = 0

    running = True

    while running:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)
        t += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_TAB:
                    show_debug = not show_debug

                elif event.key == pygame.K_r:
                    player.reset()
                    enemy.reset()
                    crystals = make_crystals()
                    projectiles.clear()
                    score = 0

        keys = pygame.key.get_pressed()

        player.update(dt, keys)
        distance, dot, can_see = enemy.update(player, dt)

        # NOVO: coleta por distância ||p_jogador - p_cristal|| < 30
        for c in crystals:
            if not c.collected:
                if np.linalg.norm(player.center - c.position(t)) < 30:
                    c.collected = True
                    score += 10

        # Vitória
        if player.rect.colliderect(goal_rect):
            score += 50
            player.reset()
            enemy.reset()
            crystals = make_crystals()
            projectiles.clear()

        # NOVO: câmera suave = combinação convexa (1-a) cam + a alvo
        target_cam = player.pos[0] - WIDTH * 0.35
        target_cam = max(0, min(target_cam, WORLD_WIDTH - WIDTH))
        alpha = 1.0 - math.exp(-8.0 * dt)
        camera_x = (1.0 - alpha) * camera_x + alpha * target_cam

        # NOVO: mouse na tela -> mundo, e vetor unitário de mira
        mx, my = pygame.mouse.get_pos()
        mouse_world = np.array([mx + camera_x, float(my)])
        aim = normalize(mouse_world - player.center)

        if pygame.mouse.get_pressed()[0] and player.shoot_cooldown <= 0.0:
            projectiles.append(Projectile(player.center, mouse_world))
            player.shoot_cooldown = 0.25

        for proj in projectiles:
            proj.update(dt)

            # Acerto por distância entre projétil e inimigo
            if enemy.alive and np.linalg.norm(proj.pos - enemy.pos) < 24:
                killed = enemy.hit(proj.vel)
                score += 25 if killed else 5
                proj.alive = False

        projectiles = [p for p in projectiles if p.alive]

        draw_background(screen, camera_x)
        draw_platforms(screen, camera_x)
        draw_goal(screen, camera_x)
        draw_crystals(screen, crystals, camera_x, t)

        if enemy.alive:
            draw_vision(screen, enemy, camera_x, distance, dot, can_see)
            draw_enemy(screen, enemy, camera_x)

        draw_projectiles(screen, projectiles, camera_x)
        draw_player(screen, player, camera_x)

        if show_debug:
            draw_debug_vectors(screen, player, enemy, camera_x, aim)

        draw_hud(screen, player, enemy, distance, dot, can_see, score, aim)
        draw_minimap(screen, player, enemy, crystals, camera_x, t)

        pygame.display.flip()

        frame += 1
        if max_frames is not None and frame >= max_frames:
            running = False

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
