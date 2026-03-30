# uv-mypy-ruff-template

パッケージマネージャに`uv`を使用し、`ruff`と`mypy`をリンター・フォーマッターとして使用するPythonプロジェクトテンプレート

## セットアップ

```bash
# 依存関係のインストール＋pre-commitフックの設定
make setup
```

これにより以下がインストールされます：
- **ruff**: 高速なPythonリンター・フォーマッター
- **mypy**: 静的型チェッカー
- **pre-commit**: Git commit時の自動チェック

## 使い方

### 基本コマンド

```bash
# リンターチェック
make lint
# または
uv run ruff check .

# コードフォーマット
make format
# または
uv run ruff format .

# 型チェック
make typecheck
# または
uv run mypy .

# リンター + 型チェック
make check

# pre-commitを全ファイルに手動実行
make precommit-run
```

### 保存時の自動フォーマット

`.vscode/settings.json`により、**Cmd+S（保存時）に自動的にruffでフォーマット**されます。

**要件**: Cursor/VSCodeに[Ruff拡張機能](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)をインストールしてください。

### Git commit時の自動チェック

`pre-commit`により、`git commit`時に以下が自動実行されます：
1. ruffによるリントチェック（自動修正）
2. ruffによるフォーマット
3. mypyによる型チェック

チェックに失敗するとcommitがブロックされます。

## 設定

### Ruff設定

`pyproject.toml`に設定を追加できます：

```toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I"]  # チェックするルール
ignore = ["E501"]         # 無視するルール
```

### Mypy設定（重要）

**現在の設定は非常に厳格（`strict = true`）です。**

すべての関数に型アノテーションが必須で、`Any`型の使用も制限されます。

#### 設定を緩くする方法

プロジェクトの段階や要件に応じて、`pyproject.toml`の`[tool.mypy]`セクションを調整できます：

```toml
[tool.mypy]
# オプション1: strictを無効化（基本的なチェックのみ）
# strict = false

# オプション2: 個別の設定を調整
strict = true
# 型アノテーションなしの関数を許可
disallow_untyped_defs = false
# 外部ライブラリの型チェックを緩く
ignore_missing_imports = true
# Anyの使用を許可
disallow_any_generics = false
```

よく使う緩和設定：
- `disallow_untyped_defs = false`: 型アノテーションなしの関数を許可
- `ignore_missing_imports = true`: 型情報がないライブラリを無視
- `disallow_untyped_calls = false`: 型なし関数の呼び出しを許可
- `warn_return_any = false`: Any型の戻り値の警告を無効化

詳細は[mypyドキュメント](https://mypy.readthedocs.io/en/stable/config_file.html)を参照してください。

## 技術スタック

- **Python**: 3.12.9+
- **パッケージマネージャ**: [uv](https://github.com/astral-sh/uv)
- **リンター/フォーマッター**: [Ruff](https://github.com/astral-sh/ruff)
- **型チェッカー**: [Mypy](https://mypy-lang.org/)
- **Git hooks**: [pre-commit](https://pre-commit.com/)
