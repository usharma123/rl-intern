import asyncio

from agent.tools.run_files import edit_run_file_handler, read_run_file_handler, write_run_file_handler


def test_run_file_tools_read_write_and_edit_inside_run_dir(tmp_path):
    run_dir = str(tmp_path)

    write_output, write_ok = asyncio.run(
        write_run_file_handler(
            {
                "run_dir": run_dir,
                "path": "train_trl.py",
                "content": "fp16=False\n",
            }
        )
    )
    edit_output, edit_ok = asyncio.run(
        edit_run_file_handler(
            {
                "run_dir": run_dir,
                "path": "train_trl.py",
                "old_str": "fp16=False",
                "new_str": "fp16=True",
            }
        )
    )
    read_output, read_ok = asyncio.run(read_run_file_handler({"run_dir": run_dir, "path": "train_trl.py"}))

    assert write_ok is True
    assert edit_ok is True
    assert read_ok is True
    assert "fp16=True" in read_output
    assert "train_trl.py" in write_output
    assert "replacements" in edit_output


def test_run_file_tools_reject_path_escape(tmp_path):
    _, ok = asyncio.run(
        write_run_file_handler(
            {
                "run_dir": str(tmp_path),
                "path": "../outside.py",
                "content": "bad",
            }
        )
    )

    assert ok is False
    assert not (tmp_path.parent / "outside.py").exists()
