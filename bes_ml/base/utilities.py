from typing import Union, Sequence
from pathlib import Path
import shutil
import subprocess


def merge_pdfs(
        inputs: Union[Sequence,list] = None,
        output: Union[str,Path] = None,
        delete_inputs: bool = False,
        verbose: bool = False,
    ):
        inputs = [Path(input) for input in inputs]
        assert len(inputs) > 0 and inputs[0].exists()
        output = Path(output)
        gs_cmd = shutil.which('gs')
        assert gs_cmd is not None, \
            "`gs` command (ghostscript) not found; available in conda-forge"
        if verbose:
            print(f"Merging PDFs into file: {output}")
        cmd = [
            gs_cmd,
            '-q',
            '-dBATCH',
            '-dNOPAUSE',
            '-sDEVICE=pdfwrite',
            '-dPDFSETTINGS=/prepress',
            '-dCompatibilityLevel=1.4',
        ]
        cmd.append(f"-sOutputFile={output.as_posix()}")
        for pdf_file in inputs:
            cmd.append(f"{pdf_file.as_posix()}")
        result = subprocess.run(cmd, check=True)
        assert result.returncode == 0 and output.exists()
        if delete_inputs is True:
            for pdf_file in inputs:
                pdf_file.unlink()
