from metacog.cli.audit import scan


def test_data_text_is_scanned_for_paths_but_not_language(tmp_path):
    path = tmp_path / "records.jsonl"
    cjk = "\u4e2d\u6587"
    private_path = "/" + "home/user/model"
    path.write_text(
        f'{{"response": "{cjk}", "path": "{private_path}"}}\n',
        encoding="utf-8",
    )
    findings = scan(tmp_path)
    assert any("absolute home path" in finding for finding in findings)
    assert not any("CJK source text" in finding for finding in findings)


def test_generic_email_addresses_are_flagged_without_institution_allowlists(tmp_path):
    path = tmp_path / "notes.md"
    email = "researcher" + "@" + "university.example"
    path.write_text(f"Contact: {email}\n", encoding="utf-8")

    findings = scan(tmp_path)

    assert any("email address" in finding for finding in findings)
