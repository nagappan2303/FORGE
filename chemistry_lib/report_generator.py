"""LaTeX emitter for species, reaction, and pathway reports. Follows the
framework of Barter et al. with this project's modifications."""

from pathlib import Path
from chemistry_lib._logging import log_message


def visualize_molecule_entry(molecule_entry, path):
    """Draw one species to a PNG with RDKit: the structure from its
    canonical SMILES, with formula and charge as the caption. Species for
    which no SMILES can be generated get a caption-only image, so the
    reports never reference a missing file."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    legend = (molecule_entry.formula.replace(" ", "")
              + "  (%+d)" % molecule_entry.charge)

    mol = None
    try:
        from chemistry_lib.reaction_questions import molgraph_to_smiles
        mg = (getattr(molecule_entry, "molecule_graph", None)
              or getattr(molecule_entry, "mol_graph", None))
        if mg is not None:
            smi = molgraph_to_smiles(mg, sanitize=True, remove_hs=True)
            if smi:
                mol = Chem.MolFromSmiles(smi)
    except Exception:
        mol = None

    drawer = rdMolDraw2D.MolDraw2DCairo(300, 250)
    if mol is not None:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, legend=legend)
    else:
        drawer.DrawMolecule(Chem.MolFromSmiles(""), legend=legend)
    drawer.FinishDrawing()
    path.write_bytes(drawer.GetDrawingText())


def visualize_molecules(mol_entries, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for index, molecule_entry in enumerate(mol_entries):
        visualize_molecule_entry(
            molecule_entry,
            folder.joinpath(str(index) + ".png"))



class ReportGenerator:

    def __init__(
            self,
            mol_entries,
            report_file_path,
            mol_pictures_folder_name='mol_pictures',
            rebuild_mol_pictures=True
    ):
        self.report_file_path = Path(report_file_path)
        self.mol_pictures_folder_name = mol_pictures_folder_name
        self.mol_pictures_folder = self.report_file_path.parent.joinpath(
            mol_pictures_folder_name)


        if rebuild_mol_pictures:
            visualize_molecules(mol_entries, self.mol_pictures_folder)

        self.mol_entries = mol_entries
        self.f = self.report_file_path.open(mode='w')


        # write in header
        self.f.write("\\documentclass{article}\n")
        self.f.write("\\usepackage{graphicx}\n")
        self.f.write("\\usepackage[margin=1cm]{geometry}\n")
        self.f.write("\\usepackage{amsmath}\n")
        self.f.write("\\pagenumbering{gobble}\n")
        self.f.write("\\begin{document}\n")
        self.f.write("\\setlength\\parindent{0pt}\n")

    def finished(self):
        self.f.write("\\end{document}")
        self.f.close()

    def emit_molecule(self, species_index, include_index=True):
        if include_index:
            self.f.write(str(species_index) + "\n")

        self.f.write(
            "\\raisebox{-.5\\height}{"
            + "\\includegraphics[width=2.6cm]{"
            + self.mol_pictures_folder_name + '/'
            + str(species_index)
            + ".png}}\n"
        )

    def emit_newline(self):
        self.f.write(
            "\n\\vspace{1cm}\n")

    def emit_newpage(self):
        self.f.write("\\newpage\n\n\n")

    def emit_verbatim(self, s):
        self.f.write('\\begin{verbatim}\n')
        self.f.write(s)
        self.f.write('\n')
        self.f.write('\\end{verbatim}\n')

    def emit_text(self,s):
        self.f.write('\n\n' + s + '\n\n')

    def emit_initial_state(self, initial_state):
        self.emit_text("initial state:")
        for species_id in initial_state:
            num = initial_state[species_id]
            if num > 0:
                self.emit_text(str(num) + " molecules of")
                self.emit_molecule(species_id)
                self.emit_newline()


    def emit_reaction(self, reaction, label=None):
        reactants_filtered = [i for i in reaction['reactants']
                              if i != -1]

        products_filtered = [i for i in reaction['products']
                             if i != -1]

        self.f.write("$$\n")
        if label is not None:
            self.f.write(label + ":  \n")

        first = True

        for reactant_index in reactants_filtered:
            if first:
                first = False
            else:
                self.f.write("+\n")

            self.emit_molecule(reactant_index)

        if 'dG' in reaction:
            self.f.write(
                "\\xrightarrow["
                + ("%.2f" % reaction["dG_barrier"]) +
                "]{" +
                ("%.2f" % reaction["dG"]) + "}\n")
        else:
            self.f.write(
                "\\xrightarrow{}\n")

        first = True
        for product_index in products_filtered:
            if first:
                first = False
            else:
                self.f.write("+\n")

            self.emit_molecule(product_index)

        self.f.write("$$")
        self.f.write("\n\n\n")

    def emit_bond_breakage(self, reaction):
        if 'reactant_bonds_broken' in reaction:
            self.f.write("reactant bonds broken:")
            for bond in reaction['reactant_bonds_broken']:
                self.emit_verbatim(str(bond))

        if 'product_bonds_broken' in reaction:
            self.f.write("product bonds broken:")
            for bond in reaction['product_bonds_broken']:
                self.emit_verbatim(str(bond))

        self.f.write("\n\n\n")
