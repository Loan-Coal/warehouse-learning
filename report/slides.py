"""Présentation PowerPoint du projet, générée depuis les mêmes chiffres et figures que le rapport.

    pip install python-pptx            # une fois (pas dans requirements.txt : outil de rédaction)
    python report/figures.py           # d'abord les figures
    python report/slides.py            # écrit report/presentation.pptx
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "presentation.pptx"

INK = RGBColor(0x1D, 0x27, 0x31)        # anthracite : fonds sombres et texte
AMBER = RGBColor(0xF2, 0xA9, 0x00)      # jaune sécurité d'entrepôt : l'accent
MUTED = RGBColor(0x5B, 0x66, 0x70)
TINT = RGBColor(0xEE, 0xF1, 0xF4)       # fond des cartes
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SHELF = RGBColor(0x8A, 0x6A, 0x4A)
DOCK = RGBColor(0xD9, 0x4F, 0x3D)
PPO_GREEN = RGBColor(0x1B, 0xAF, 0x7A)
FONT = "Calibri"

W, H = 13.333, 7.5
MARGIN = 0.6


def rgb_fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, {options})."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paragraphs = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(6)
        for piece in ([(para, {})] if isinstance(para, str) else para):
            t, opt = piece
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = FONT
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.italic = opt.get("italic", False)
            f.color.rgb = opt.get("color", color)
    return box


def bullets(slide, x, y, w, h, items, size=16, color=INK):
    box = text(slide, x, y, w, h, [""], size=size, color=color)
    tf = box.text_frame
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(8)
        pieces = [(item, {})] if isinstance(item, str) else item
        r0 = p.add_run()
        r0.text = "▪  "
        r0.font.color.rgb = AMBER
        r0.font.size = Pt(size)
        r0.font.name = FONT
        for t, opt in pieces:
            r = p.add_run()
            r.text = t
            r.font.name = FONT
            r.font.size = Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", False)
            r.font.color.rgb = opt.get("color", color)
    return box


def card(slide, x, y, w, h, fill=TINT):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.adjustments[0] = 0.06
    rgb_fill(shape, fill)
    return shape


def badge(slide, x, y, label, d=0.5, fill=AMBER, color=INK, size=16):
    shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    rgb_fill(shape, fill)
    tf = shape.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(size), True, color
    return shape


def picture(slide, name, x, y, w=None, h=None):
    path = FIG / f"{name}.png"
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def background(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def content_slide(prs, title, notes, kicker=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, WHITE)
    if kicker:
        text(s, MARGIN, 0.42, 9, 0.3, kicker.upper(), size=12, color=MUTED, bold=True)
    text(s, MARGIN, 0.7, W - 2 * MARGIN, 0.8, title, size=30, bold=True, name="Titre")
    s.notes_slide.notes_text_frame.text = notes
    return s


def number_footer(slide, n, dark=False):
    text(slide, W - MARGIN - 1, H - 0.45, 1, 0.3, str(n), size=11, color=WHITE if dark else MUTED,
         align=PP_ALIGN.RIGHT)


def mini_map(slide, x, y, cell, layout, robots=(), color_robot=AMBER):
    rows = [r for r in layout.strip().splitlines()]
    for j, row in enumerate(rows):
        for i, ch in enumerate(row):
            sq = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + i * cell), Inches(y + j * cell),
                                        Inches(cell * 0.94), Inches(cell * 0.94))
            rgb_fill(sq, {"S": SHELF, "D": DOCK}.get(ch, RGBColor(0x2B, 0x38, 0x45)))
    for (i, j) in robots:
        d = cell * 0.62
        o = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + i * cell + (cell * 0.94 - d) / 2),
                                   Inches(y + j * cell + (cell * 0.94 - d) / 2), Inches(d), Inches(d))
        rgb_fill(o, color_robot)


MAZE = """
...S.......
.S.S.S.S.S.
.S.......S.
.SSS.S.S.S.
...S...S...
SS.S.SSSSS.
.S...S...S.
.S.SSS.S.S.
...D...D...
"""


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    n = 0

    # 1. Titre -------------------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, INK)
    mini_map(s, 8.3, 1.55, 0.42, MAZE, robots=[(0, 0), (4, 2), (8, 4), (10, 8)])
    text(s, MARGIN, 1.7, 7.2, 0.4, "PROJET D'APPRENTISSAGE PAR RENFORCEMENT", size=13, color=AMBER, bold=True)
    text(s, MARGIN, 2.2, 7.3, 2.0, "Coordonner une flotte de robots d'entrepôt", size=40, color=WHITE, bold=True)
    text(s, MARGIN, 4.35, 7.2, 0.9, "Du Q-learning tabulaire au PPO centralisé", size=20, color=WHITE)
    text(s, MARGIN, 5.75, 7.2, 0.9, ["Antoine de Chabannes · Lohann Colle · Juliette Girardin · Loïc",
                                      "Master 2 · Université Côte d'Azur"], size=14, color=RGBColor(0xC9, 0xD1, 0xD8))
    s.notes_slide.notes_text_frame.text = (
        "Présenter l'équipe et le sujet : une flotte de chariots élévateurs dans un entrepôt en grille. "
        "Question centrale : quelles méthodes d'apprentissage par renforcement passent à l'échelle quand la flotte grandit ?")

    # 2. Le problème -------------------------------------------------------------------------
    n += 1
    s = content_slide(prs, "Livrer sans se bloquer dans des allées d'une case",
                      "Chaque robot reçoit une mission : prélever une caisse sur une étagère pour l'amener à un quai, ou l'inverse. "
                      "Les allées font une case de large : deux robots face à face se bloquent. "
                      "Architecture : la file de tâches affecte les missions (comme un WMS), chaque robot choisit ses déplacements.",
                      kicker="Le problème")
    mini_map(s, MARGIN, 1.9, 0.44, MAZE, robots=[(2, 0), (4, 2), (6, 4), (10, 6)])
    text(s, MARGIN, 6.05, 5, 0.6, [[("■ ", {"color": SHELF}), ("étagère   ", {}), ("■ ", {"color": DOCK}),
                                    ("quai   ", {}), ("● ", {"color": AMBER}), ("robot", {})]], size=12, color=MUTED)
    items = [("Prélever", "étagère → quai, ou ranger : quai → étagère"),
             ("Naviguer", "labyrinthe nouveau à chaque épisode, humains qui errent"),
             ("Se coordonner", "allées d'une case : deux robots face à face se bloquent")]
    for i, (head, body) in enumerate(items):
        y = 2.0 + i * 1.35
        badge(s, 6.35, y, str(i + 1))
        text(s, 7.1, y - 0.02, 5.6, 0.4, head, size=20, bold=True)
        text(s, 7.1, y + 0.42, 5.6, 0.7, body, size=16, color=MUTED)
    number_footer(s, n + 1)

    # 3. L'environnement ---------------------------------------------------------------------
    n += 1
    s = content_slide(prs, "Un environnement à difficulté réglable",
                      "Six actions : prendre, déposer, tourner à droite ou à gauche, avancer, reculer. Prendre exige de faire face à la cible. "
                      "Récompense : +10 par livraison, -0,05 par pas, -1 si bloqué, -0,1 si action impossible ; moitié propre, moitié moyenne de la flotte. "
                      "Tous les facteurs sont désactivés par défaut : on les active un par un.",
                      kicker="L'environnement")
    cards = [("Actions", ["6 actions discrètes", "prendre / déposer face à la cible", "pas d'attente : on tourne sur place"]),
             ("Observation", ["position, orientation, mission", "4 rayons : mur, étagère, robot, humain", "les autres ne sont vus que par les rayons"]),
             ("Récompense", ["+10 par livraison", "−0,05 par pas, −1 si bloqué", "50 % propre, 50 % moyenne de flotte"])]
    cw = (W - 2 * MARGIN - 2 * 0.35) / 3
    for i, (head, lines) in enumerate(cards):
        x = MARGIN + i * (cw + 0.35)
        card(s, x, 1.85, cw, 2.65)
        text(s, x + 0.3, 2.05, cw - 0.6, 0.45, head, size=20, bold=True)
        bullets(s, x + 0.3, 2.65, cw - 0.6, 1.8, lines, size=15)
    text(s, MARGIN, 4.85, 6, 0.4, "Scénarios de référence", size=18, bold=True)
    rows = [("basic", "carte 7×6 fixe, sans facteur"),
            ("hard", "labyrinthe 11×9, 4 robots, 2 humains, priorités, robots lents"),
            ("big", "labyrinthe 21×21, 6 robots, 3 humains, tout activé, 400 pas")]
    for i, (name, desc) in enumerate(rows):
        y = 5.35 + i * 0.5
        text(s, MARGIN, y, 1.2, 0.4, name, size=16, bold=True, color=INK)
        text(s, MARGIN + 1.3, y, 10.5, 0.4, desc, size=16, color=MUTED)
    number_footer(s, n + 1)

    # 4. Mesurer honnêtement -----------------------------------------------------------------
    n += 1
    s = content_slide(prs, "Une courbe d'entraînement n'est pas un résultat",
                      "Notre première « courbe d'apprentissage » ne mesurait que la décroissance d'un professeur glouton : le réseau n'avait rien appris en 3000 épisodes. "
                      "Depuis : évaluation sur cartes jamais vues, disjointes par construction, et 100 épisodes par chiffre.",
                      kicker="Protocole")
    stats = [("100", "épisodes par chiffre rapporté", "à 20, le glouton passe de 5,00 à 6,23 : bruit"),
             ("0", "carte commune entre entraînement et évaluation", "le lanceur refuse de démarrer sinon"),
             ("3000", "épisodes « appris » par erreur", "c'était le professeur qui livrait")]
    for i, (big, label, sub) in enumerate(stats):
        x = MARGIN + i * (cw + 0.35)
        card(s, x, 2.0, cw, 3.4)
        text(s, x + 0.35, 2.25, cw - 0.7, 1.2, big, size=60, bold=True, color=AMBER)
        text(s, x + 0.35, 3.55, cw - 0.7, 1.0, label, size=18, bold=True)
        text(s, x + 0.35, 4.6, cw - 0.7, 0.8, sub, size=14, color=MUTED)
    text(s, MARGIN, 5.85, W - 2 * MARGIN, 0.8,
         "On rapporte d'abord les livraisons et les actions invalides, puis les blocages ; le retour moyen a un plancher indépendant de la politique.",
         size=15, color=MUTED)
    number_footer(s, n + 1)

    # 5. Méthodes ----------------------------------------------------------------------------
    n += 1
    s = content_slide(prs, "Cinq façons de décider, du tabulaire au planificateur",
                      "Le glouton est la barre à battre. La Q-table centralisée et les apprenants indépendants sont les méthodes tabulaires. "
                      "PPO est notre méthode principale. Le planificateur et la recherche ne s'entraînent pas : ils bornent ce qu'apporte l'information.",
                      kicker="Méthodes")
    methods = [("Glouton", "BFS à chaque pas, ε = 0,1", "référence", MUTED),
               ("Q-table centralisée", "une table, 6ⁿ actions jointes", "tabulaire", AMBER),
               ("Q-learning indépendant", "une table par robot", "tabulaire", AMBER),
               ("PPO centralisé", "masqué, encodeur par paires", "profond", PPO_GREEN),
               ("Planificateur", "réservations espace-temps", "non appris", MUTED)]
    mw = (W - 2 * MARGIN - 4 * 0.25) / 5
    for i, (head, body, tag, col) in enumerate(methods):
        x = MARGIN + i * (mw + 0.25)
        card(s, x, 2.0, mw, 3.3)
        badge(s, x + 0.25, 2.25, str(i + 1), fill=col, color=WHITE if col != AMBER else INK)
        text(s, x + 0.25, 3.0, mw - 0.4, 0.9, head, size=18, bold=True)
        text(s, x + 0.25, 3.95, mw - 0.4, 0.8, body, size=14, color=MUTED)
        text(s, x + 0.25, 4.8, mw - 0.4, 0.3, tag.upper(), size=11, bold=True, color=MUTED)
    text(s, MARGIN, 5.75, W - 2 * MARGIN, 0.8,
         "Toutes les méthodes passent par la même interface et les mêmes tests (183) : seul l'algorithme change.",
         size=15, color=MUTED)
    number_footer(s, n + 1)

    # 6. Q-table : l'explosion ---------------------------------------------------------------
    n += 1
    s = content_slide(prs, "Q-table centralisée : la taille explose avec la flotte",
                      "Un seul contrôleur choisit l'action de tous les robots à la fois. "
                      "État : position, orientation et mission de chaque robot ; 34 cases × 4 orientations × 17 situations de mission par robot. "
                      "Mise à jour Q-learning sur la somme des récompenses, ε décroissant de 1 à 0,05, plafond de 600 Mo.",
                      kicker="Q-table centralisée")
    picture(s, "states", MARGIN, 1.75, w=7.4)
    cols = [("2 robots", "36", "5,4·10⁶"), ("3 robots", "216", "1,2·10¹⁰"), ("4 robots", "1 296", "2,9·10¹³")]
    text(s, 8.55, 1.85, 4.2, 0.4, [[("actions jointes  ", {"bold": True}), ("·  états théoriques", {"color": MUTED})]], size=14)
    for i, (lab, a, st) in enumerate(cols):
        y = 2.45 + i * 1.05
        card(s, 8.55, y, 4.2, 0.85)
        text(s, 8.8, y + 0.22, 1.3, 0.4, lab, size=14, color=MUTED)
        text(s, 10.05, y + 0.12, 1.2, 0.6, a, size=24, bold=True)
        text(s, 11.25, y + 0.2, 1.45, 0.5, st, size=16, color=MUTED)
    text(s, 8.55, 5.75, 4.2, 1.0, "6ⁿ actions × (34·4·17)ⁿ états : la mémoire sature avant l'apprentissage.",
         size=16, bold=True)
    number_footer(s, n + 1)

    # 7. Q-table : courbes -------------------------------------------------------------------
    n += 1
    s = content_slide(prs, "À 3 robots, la table est pleine en un million de pas",
                      "À 2 robots, la table a vu tous les états d'évaluation mais ne livre qu'une caisse par épisode après 57 millions de pas ; elle progressait encore. "
                      "À 3 et 4 robots, le plafond est atteint presque immédiatement, 98 à 99,6 % des mises à jour sont rejetées, et 99,5 à 100 % des états d'évaluation sont inconnus.",
                      kicker="Q-table centralisée")
    picture(s, "q_curves", MARGIN, 1.7, w=8.3)
    bullets(s, 9.25, 1.95, 3.5, 4.6, [
        [("2 robots : ", {"bold": True}), ("1,2 livraison/épisode après 57 M de pas", {})],
        [("3–4 robots : ", {"bold": True}), ("plafond 600 Mo atteint, 98–99,6 % des mises à jour rejetées", {})],
        [("En évaluation : ", {"bold": True}), ("99,5–100 % des états jamais vus", {})],
    ], size=16)
    number_footer(s, n + 1)

    # 8. Apprenants indépendants (Loïc et Juliette) ------------------------------------------
    n += 1
    s = content_slide(prs, "Q-learning indépendant : un cerveau par robot",
                      "Slide à compléter par Loïc et Juliette.",
                      kicker="Apprenants indépendants")
    card(s, MARGIN, 1.9, W - 2 * MARGIN, 4.6)
    text(s, MARGIN + 0.5, 3.6, W - 2 * MARGIN - 1, 1.2,
         "À compléter (Loïc et Juliette) : état, mise à jour, non-stationnarité, résultats",
         size=20, color=MUTED, align=PP_ALIGN.CENTER)
    number_footer(s, n + 1)

    # 9. PPO : ce qui l'a fait marcher -------------------------------------------------------
    n += 1
    s = content_slide(prs, "PPO : quatre choix mesurés font la différence",
                      "On-policy car les robots apprennent ensemble : une mémoire de rejeu contiendrait la dynamique d'anciens coéquipiers, et mélangerait les cartes. "
                      "Un seul réseau lit toute la flotte : un encodeur par paires compare le robot i à chaque robot j, agrégé par max et moyenne. "
                      "Le masquage a fait passer les actions invalides de 1128 à 0 par épisode ; les distances BFS remplacent 882 dimensions de carte par 5 ; "
                      "la forme textbook du shaping faisait flâner l'agent ; l'échantillonnage brise les face-à-face.",
                      kicker="PPO centralisé")
    levers = [("Masquer les actions", "1128 → 0", "actions invalides par épisode"),
              ("Distances BFS", "882 → 5", "dimensions d'entrée, et la carte n'entre plus dans le réseau"),
              ("Signal de collision", "45,2 → 12,6", "blocages par épisode"),
              ("Échantillonner", "211 → ≈ 0", "blocages par interblocage miroir")]
    lw = (W - 2 * MARGIN - 0.35) / 2
    for i, (head, num, sub) in enumerate(levers):
        x = MARGIN + (i % 2) * (lw + 0.35)
        y = 1.85 + (i // 2) * 1.75
        card(s, x, y, lw, 1.5)
        text(s, x + 0.35, y + 0.2, 2.75, 0.4, head, size=18, bold=True)
        text(s, x + 0.35, y + 0.7, 2.75, 0.7, sub, size=14, color=MUTED)
        text(s, x + 3.1, y + 0.35, lw - 3.4, 0.9, num, size=26, bold=True, color=PPO_GREEN, align=PP_ALIGN.RIGHT)
    text(s, MARGIN, 5.55, W - 2 * MARGIN, 1.2,
         [[("Architecture : ", {"bold": True}),
           ("un seul réseau lit toute la flotte ; pour chaque robot, un encodeur compare chaque paire (i, j), "
            "agrégée par max (« une menace ? ») et moyenne (« encombré ? »). On-policy, sans schedule.", {"color": MUTED})]],
         size=15)
    number_footer(s, n + 1)

    # 10. Passage à l'échelle ----------------------------------------------------------------
    n += 1
    s = content_slide(prs, "PPO bat le glouton partout, la Q-table s'effondre",
                      "Carte 7×6, 100 épisodes d'évaluation à graines réservées. "
                      "PPO : +50 % sur le glouton à 2 robots, +74 % à 4. La Q-table livre 1,2 caisse à 2 robots et rien au-delà. "
                      "Le glouton plafonne vers 10 livraisons car il passe son temps à contourner.",
                      kicker="Résultats")
    picture(s, "scaling", MARGIN, 1.7, w=8.2)
    for i, (big, lab) in enumerate((("+50 %", "PPO vs glouton, 2 robots"), ("+74 %", "PPO vs glouton, 4 robots"),
                                    ("0", "livraison Q-table dès 3 robots"))):
        y = 1.9 + i * 1.55
        text(s, 9.3, y, 3.4, 0.8, big, size=40, bold=True, color=PPO_GREEN if i < 2 else AMBER)
        text(s, 9.3, y + 0.8, 3.4, 0.5, lab, size=15, color=MUTED)
    number_footer(s, n + 1)

    # 11. Courbes PPO ------------------------------------------------------------------------
    n += 1
    s = content_slide(prs, "PPO atteint son plateau en moins d'un million de pas",
                      "Un même réseau traite n'importe quel nombre de robots, car ses entrées ne dépendent ni de la carte ni de la taille de la flotte. "
                      "Il dépasse le glouton (pointillés) dès les premières centaines de milliers de pas.",
                      kicker="Résultats")
    picture(s, "ppo_curves", MARGIN, 1.7, w=8.7)
    bullets(s, 9.6, 2.0, 3.15, 4.4, [
        "Entrées indépendantes de la carte et de n : généralise aux états jamais vus",
        "60 min par taille, ≈ 1 400 pas/s sur CPU",
        "Pointillés : glouton avec la même flotte",
    ], size=16)
    number_footer(s, n + 1)

    # 12. Scénario big -----------------------------------------------------------------------
    n += 1
    s = content_slide(prs, "Sur le scénario le plus dur, PPO livre 73 % de plus",
                      "Labyrinthe 21×21 nouveau à chaque épisode, 6 robots hétérogènes, 3 humains, priorités, 400 pas ; 100 labyrinthes jamais vus. "
                      "Le glouton oscille : un tiers de ses déplacements reviennent en arrière. "
                      "Les demi-tours de PPO surviennent à 98,5 % près d'un autre agent : c'est sa façon d'attendre.",
                      kicker="Résultats · scénario big")
    hdr = ["", "Livraisons", "Job ratio", "Retours arrière"]
    data = [["Glouton", "10,6", "2,12", "0,340"], ["PPO (12 h)", "18,3", "1,50", "0,081"]]
    tbl = s.shapes.add_table(3, 4, Inches(MARGIN), Inches(1.95), Inches(7.6), Inches(1.8)).table
    for c, wdt in enumerate((2.2, 1.8, 1.8, 1.8)):
        tbl.columns[c].width = Inches(wdt)
    for r, row in enumerate([hdr] + data):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = INK if r == 0 else (TINT if r == 1 else WHITE)
            tf = cell.text_frame
            tf.paragraphs[0].alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            run = tf.paragraphs[0].add_run()
            run.text = val
            run.font.name, run.font.size = FONT, Pt(18 if r else 15)
            run.font.bold = r == 0 or (r == 2 and c > 0)
            run.font.color.rgb = WHITE if r == 0 else (PPO_GREEN if r == 2 and c > 0 else INK)
    bullets(s, MARGIN, 4.3, 7.6, 2.4, [
        [("Le glouton oscille : ", {"bold": True}), ("un tiers de ses pas reviennent sur la case quittée", {})],
        [("PPO attend en tournant : ", {"bold": True}), ("98,5 % de ses demi-tours sont près d'un autre agent", {})],
        [("Une action WAIT ", {"bold": True}), ("accélère l'apprentissage mais crée des interblocages en évaluation", {})],
    ], size=16)
    card(s, 8.75, 1.95, 4.0, 4.6, fill=INK)
    text(s, 9.1, 2.3, 3.3, 1.0, "18,3", size=60, bold=True, color=PPO_GREEN)
    text(s, 9.1, 3.45, 3.3, 0.6, "livraisons par épisode", size=16, color=WHITE)
    text(s, 9.1, 4.3, 3.3, 1.0, "10,6", size=40, bold=True, color=RGBColor(0xC9, 0xD1, 0xD8))
    text(s, 9.1, 5.2, 3.3, 0.6, "pour le glouton", size=16, color=RGBColor(0xC9, 0xD1, 0xD8))
    number_footer(s, n + 1)

    # 13. Échelle d'information --------------------------------------------------------------
    n += 1
    s = content_slide(prs, "Ce que vaut l'information face à l'apprentissage",
                      "Le planificateur réserve les cases dans l'espace-temps : les robots ne se bloquent jamais entre eux. "
                      "Trois niveaux : ce que voit PPO, la position de tous les humains, et le futur réel des humains. "
                      "À compléter avec les chiffres de search (Lohann).",
                      kicker="Résultats · planificateur")
    if (FIG / "ladder.png").is_file():
        picture(s, "ladder", MARGIN, 1.7, w=8.0)
    bullets(s, 9.0, 1.95, 3.75, 4.6, [
        [("obs : ", {"bold": True}), ("ce que voit PPO", {})],
        [("state : ", {"bold": True}), ("position de tous les humains", {})],
        [("future : ", {"bold": True}), ("où les humains vont aller", {})],
        [("search : ", {"bold": True}), ("PPO + simulations à 6 pas, à compléter (Lohann)", {"color": MUTED})],
    ], size=16)
    number_footer(s, n + 1)

    # 14. Conclusion -------------------------------------------------------------------------
    n += 1
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, INK)
    text(s, MARGIN, 0.9, W - 2 * MARGIN, 0.9, "Ce qu'on retient", size=36, bold=True, color=WHITE)
    take = [("Le tabulaire centralisé ne passe pas à l'échelle",
             "la mémoire sature avant l'apprentissage ; aveugle sur 99,5 % des états dès 3 robots"),
            ("PPO généralise",
             "entrées indépendantes de la carte et de la flotte ; +74 % sur le glouton à 4 robots, 18,3 contre 10,6 sur big"),
            ("Les choix mesurés comptent plus que l'algorithme",
             "masquer, donner les distances BFS, façonner sans flânerie, évaluer sur cartes inédites")]
    for i, (head, body) in enumerate(take):
        y = 2.2 + i * 1.45
        badge(s, MARGIN, y, str(i + 1))
        text(s, MARGIN + 0.8, y - 0.03, 11.2, 0.5, head, size=22, bold=True, color=WHITE)
        text(s, MARGIN + 0.8, y + 0.5, 11.2, 0.6, body, size=16, color=RGBColor(0xC9, 0xD1, 0xD8))
    text(s, MARGIN, 6.6, W - 2 * MARGIN, 0.4, "Pistes : VDN / QMIX, têtes autorégressives, caractéristiques vectorisées",
         size=14, color=AMBER)
    s.notes_slide.notes_text_frame.text = (
        "Trois messages. Puis ouvrir les questions ; démo possible : python -m isaac.demo --checkpoint checkpoints/ppo_4.")
    number_footer(s, n + 1, dark=True)

    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
