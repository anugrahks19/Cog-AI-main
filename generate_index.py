import os

def generate_tree(startpath):
    tree_str = f"# Project Index\n\n```text\n{os.path.basename(startpath)}/\n"
    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', 'dist', 'build', '.venv', '.gemini', '.idea', '.vscode', 'public', 'assets']]
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        if level > 0:
            tree_str += f"{indent}|-- {os.path.basename(root)}/\n"
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if f in ['project_index.md', 'project_index.txt', 'generate_index.py', 'bun.lockb', 'package-lock.json', 'alzheimers_disease_data.csv']:
                continue
            tree_str += f"{subindent}|-- {f}\n"
    tree_str += "```\n"
    return tree_str

if __name__ == '__main__':
    with open('project_index.md', 'w', encoding='utf-8') as f:
        f.write(generate_tree('.'))
