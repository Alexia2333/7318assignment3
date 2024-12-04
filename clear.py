file_path = "utils/data_loader.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 移除不可见字符
cleaned_content = ''.join(c for c in content if c.isprintable() or c.isspace())

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(cleaned_content)

print("File cleaned of non-printable characters.")
