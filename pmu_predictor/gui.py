"""Interface graphique (Tkinter, aucune dépendance externe)."""

from __future__ import annotations

import csv
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from . import config as C
from .loader import LoadedRace, arrivee_path, evaluate, load_race, parse_arrivee
from .model import Prediction, format_prediction, predict

COLUMNS = [
    ("rang", "Rang", 50), ("num", "N°", 45), ("nom", "Cheval", 180), ("driver", "Driver", 150),
    ("cote", "Cote", 60), ("musique", "Musique", 170), ("score", "Score", 70),
    ("pm", "Proba modèle", 95), ("pk", "Proba marché", 95), ("val", "Valeur", 65),
]


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PMU Predictor — pronostics de trot")
        self.geometry("1280x760")
        self.minsize(1000, 600)

        self.loaded: list[LoadedRace] = []
        self.paths: list[str] = []
        self.predictions: list[Prediction] = []
        self.weights: dict[str, float] = dict(C.DEFAULT_WEIGHTS)
        self.sliders: dict[str, tuple[tk.DoubleVar, ttk.Label]] = {}

        ttk.Style(self).theme_use("clam")
        self._build_toolbar()
        self._build_body()
        self.status = tk.StringVar(value="Charge un ou plusieurs fichiers JSON de course.")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken").pack(fill="x", side="bottom")

    # ------------------------------------------------------------------ UI
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=6)
        bar.pack(fill="x")
        ttk.Button(bar, text="Charger fichier(s)…", command=self.open_files).pack(side="left", padx=2)
        ttk.Button(bar, text="Charger dossier…", command=self.open_folder).pack(side="left", padx=2)
        ttk.Button(bar, text="Exporter CSV", command=self.export_csv).pack(side="left", padx=2)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(bar, text="Charger poids…", command=self.load_weights).pack(side="left", padx=2)
        ttk.Button(bar, text="Enregistrer poids…", command=self.save_weights).pack(side="left", padx=2)
        ttk.Button(bar, text="Réinitialiser poids", command=self.reset_weights).pack(side="left", padx=2)

    def _build_body(self) -> None:
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # colonne gauche : liste des courses
        left = ttk.Frame(paned, padding=4)
        ttk.Label(left, text="Courses chargées").pack(anchor="w")
        self.race_list = tk.Listbox(left, exportselection=False, width=26)
        self.race_list.pack(fill="both", expand=True)
        self.race_list.bind("<<ListboxSelect>>", lambda _e: self.show_selected_race())
        paned.add(left, weight=0)

        # centre : tableau + détail
        center = ttk.Frame(paned, padding=4)
        self.header = tk.StringVar(value="")
        ttk.Label(center, textvariable=self.header, font=("TkDefaultFont", 11, "bold")).pack(anchor="w")

        self.tree = ttk.Treeview(center, columns=[c[0] for c in COLUMNS], show="headings", height=16)
        for key, label, width in COLUMNS:
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor="center" if key not in ("nom", "driver", "musique") else "w")
        self.tree.tag_configure("top5", background="#dff3e1")
        self.tree.tag_configure("outsider", background="#ffe8c2")
        self.tree.pack(fill="both", expand=True, pady=4)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.show_horse_detail())

        btns = ttk.Frame(center)
        btns.pack(fill="x")
        self.verify_btn = ttk.Button(btns, text="Vérifier avec le résultat réel", command=self.verify, state="disabled")
        self.verify_btn.pack(side="left")
        ttk.Button(btns, text="Saisir l'arrivée…", command=self.enter_result).pack(side="left", padx=4)
        ttk.Label(btns, text="  (le résultat n'est lu qu'après la prédiction, jamais utilisé par le modèle)",
                  foreground="#666").pack(side="left")

        self.detail = tk.Text(center, height=11, wrap="word", font=("TkFixedFont", 10))
        self.detail.pack(fill="both", expand=False, pady=4)
        paned.add(center, weight=1)

        # droite : poids
        right = ttk.Frame(paned, padding=6)
        ttk.Label(right, text="Poids du modèle", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        for key, (default, label, helptext) in C.FEATURES.items():
            row = ttk.Frame(right)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=label, width=26).pack(side="left")
            var = tk.DoubleVar(value=default)
            val = ttk.Label(row, text=f"{default:.2f}", width=5)
            scale = ttk.Scale(row, from_=0.0, to=2.0, variable=var, length=110,
                              command=lambda _v, k=key: self.on_weight(k))
            scale.pack(side="left")
            val.pack(side="left")
            self.sliders[key] = (var, val)
            _Tip(scale, helptext)
        paned.add(right, weight=0)

    # -------------------------------------------------------------- chargement
    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(title="Fichiers de course",
                                            filetypes=[("Courses", "*.json *.txt"), ("Tous", "*.*")])
        self.add_files(paths)

    def open_folder(self) -> None:
        folder = filedialog.askdirectory(title="Dossier de courses")
        if folder:
            files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".json", ".txt"))
            self.add_files([str(p) for p in files])

    def add_files(self, paths) -> None:
        errors = []
        for p in paths:
            try:
                self.loaded.append(load_race(p))
                self.paths.append(str(p))
            except Exception as exc:
                errors.append(f"{Path(p).name} : {exc}")
        if errors:
            messagebox.showwarning("Fichiers ignorés", "\n".join(errors))
        self.refresh()
        if self.loaded:
            self.race_list.selection_clear(0, "end")
            self.race_list.selection_set("end")
            self.show_selected_race()

    # ------------------------------------------------------------------ calcul
    def refresh(self) -> None:
        self.predictions = []
        self.race_list.delete(0, "end")
        for lr in self.loaded:
            try:
                self.predictions.append(predict(lr.race, self.weights))
            except Exception as exc:
                messagebox.showerror("Erreur", f"{lr.race.name} : {exc}")
                self.predictions.append(None)  # type: ignore[arg-type]
            self.race_list.insert("end", lr.race.name)
        sel = self.race_list.curselection()
        if sel:
            self.show_selected_race()

    def on_weight(self, key: str) -> None:
        var, label = self.sliders[key]
        self.weights[key] = round(var.get(), 2)
        label.config(text=f"{self.weights[key]:.2f}")
        self.refresh_keep_selection()

    def refresh_keep_selection(self) -> None:
        sel = self.race_list.curselection()
        idx = sel[0] if sel else None
        self.refresh()
        if idx is not None and idx < len(self.loaded):
            self.race_list.selection_set(idx)
            self.show_selected_race()

    def _current(self) -> int | None:
        sel = self.race_list.curselection()
        return sel[0] if sel else None

    # ------------------------------------------------------------------ affichage
    def show_selected_race(self) -> None:
        i = self._current()
        self.tree.delete(*self.tree.get_children())
        self.detail.delete("1.0", "end")
        if i is None or i >= len(self.predictions) or self.predictions[i] is None:
            return
        pred, lr = self.predictions[i], self.loaded[i]
        self.header.set(f"{pred.race_name} — {len(pred.horses)} partants — course {pred.openness}")
        top5_nums = {h.num for h in pred.top5}
        for h in pred.horses:
            tags = ("top5",) if h.num in top5_nums else ()
            if pred.outsider and h.num == pred.outsider.num:
                tags = ("outsider",)
            self.tree.insert("", "end", iid=str(h.num), tags=tags, values=(
                h.rank, h.num, h.nom, h.driver, f"{h.odds:g}" if h.odds else "n/c", h.musique,
                f"{h.score:+.2f}", f"{h.prob_model:.1%}",
                f"{h.prob_market:.1%}" if h.prob_market else "—",
                f"{h.value_ratio:.2f}" if h.value_ratio else "—"))
        self.verify_btn.config(state="normal" if lr.result else "disabled")
        self.detail.insert("1.0", format_prediction(pred) + "\n\n(Vert = top 5 · Orange = outsider à surveiller)")

    def show_horse_detail(self) -> None:
        i = self._current()
        sel = self.tree.selection()
        if i is None or not sel:
            return
        h = next(x for x in self.predictions[i].horses if str(x.num) == sel[0])
        pos, neg = h.reasons(4)
        lines = [f"n°{h.num} {h.nom}  —  rang {h.rank}  —  driver {h.driver}",
                 f"Musique : {h.musique}   Cote : {h.odds if h.odds else 'n/c'}",
                 f"Proba modèle {h.prob_model:.1%}"
                 + (f" · marché {h.prob_market:.1%}" if h.prob_market else ""),
                 "", "Points forts : " + ("; ".join(pos) or "—"),
                 "Points faibles : " + ("; ".join(neg) or "—"), "", "Contributions au score :"]
        for k, v in sorted(h.contributions.items(), key=lambda kv: -abs(kv[1])):
            lines.append(f"  {C.FEATURES[k][1]:<30} {v:+.2f}")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", "\n".join(lines))

    # ------------------------------------------------------------------ vérification
    def verify(self) -> None:
        i = self._current()
        if i is None or not self.loaded[i].result:
            return
        pred, res = self.predictions[i], self.loaded[i].result
        names = {h.num: h.nom for h in pred.horses}
        ev = evaluate([h.num for h in pred.top5], res)
        arr = " – ".join(f"{n} ({names.get(n, '?')})" for n in ev["arrivee_top5"])
        txt = [f"Arrivée réelle : {arr}",
               f"Pronostic     : {' – '.join(str(n) for n in ev['pronostic_top5'])}", "",
               f"Bons chevaux dans le top 5 : {ev['bons_dans_top5']}/5",
               f"Positions exactes          : {ev['positions_exactes']}/5",
               f"Gagnant trouvé             : {'oui' if ev['gagnant_trouve'] else 'non'}",
               f"Tiercé désordre            : {'oui' if ev['tierce_desordre'] else 'non'}",
               f"Quinté désordre            : {'oui' if ev['quinte_desordre'] else 'non'}",
               f"Quinté ordre               : {'oui' if ev['quinte_ordre'] else 'non'}"]
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", "\n".join(txt))

    def enter_result(self) -> None:
        i = self._current()
        if i is None:
            return
        txt = simpledialog.askstring("Arrivée", "Numéros dans l'ordre d'arrivée (ex. 7-9-2-3-13) :", parent=self)
        if not txt:
            return
        try:
            self.loaded[i].result = parse_arrivee(txt)
        except ValueError as exc:
            messagebox.showerror("Arrivée", str(exc))
            return
        arrivee_path(self.paths[i]).write_text("-".join(str(n) for n in sorted(
            self.loaded[i].result, key=self.loaded[i].result.get)), encoding="utf-8")
        self.verify_btn.config(state="normal")
        self.verify()

    # ------------------------------------------------------------------ export / poids
    def export_csv(self) -> None:
        if not self.predictions:
            messagebox.showinfo("Export", "Rien à exporter.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["course", "rang", "num", "cheval", "driver", "cote", "score", "proba_modele", "proba_marche"])
            for pred in self.predictions:
                if pred is None:
                    continue
                for h in pred.horses:
                    w.writerow([pred.race_name, h.rank, h.num, h.nom, h.driver, h.odds or "",
                                f"{h.score:.3f}", f"{h.prob_model:.4f}",
                                f"{h.prob_market:.4f}" if h.prob_market else ""])
        self.status.set(f"Export enregistré : {path}")

    def load_weights(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
            self.set_weights({k: float(v) for k, v in data.items() if k in C.FEATURES})
        except Exception as exc:
            messagebox.showerror("Poids", str(exc))

    def save_weights(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile="weights.json")
        if path:
            Path(path).write_text(json.dumps(self.weights, indent=2))
            self.status.set(f"Poids enregistrés : {path}")

    def reset_weights(self) -> None:
        self.set_weights(dict(C.DEFAULT_WEIGHTS))

    def set_weights(self, new: dict[str, float]) -> None:
        self.weights.update(new)
        for k, (var, label) in self.sliders.items():
            var.set(self.weights[k])
            label.config(text=f"{self.weights[k]:.2f}")
        self.refresh_keep_selection()


class _Tip:
    """Infobulle minimale."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget, self.text, self.win = widget, text, None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _e=None) -> None:
        x = self.widget.winfo_rootx() - 260
        y = self.widget.winfo_rooty() + 24
        self.win = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, background="#ffffe0", relief="solid", borderwidth=1,
                 wraplength=300, justify="left").pack()

    def hide(self, _e=None) -> None:
        if self.win:
            self.win.destroy()
            self.win = None


def run() -> None:
    App().mainloop()
