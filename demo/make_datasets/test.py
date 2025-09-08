# 1. 导入所需库
from pathlib import Path
import json
from datasets import load_dataset
import sys  # 导入 sys 库用于退出

# --- 配置 ---
# 使用 Path.cwd() 确保路径基于当前工作目录，简单可靠




root = Path().absolute().parent.parent
root = str(root)
NUM_SAMPLES = 4000
OUTPUT_JSON_FILE = f"{root}/data/input/sft_dataset_{NUM_SAMPLES}.json"

def process_example(example):
    """处理单个数据样本，将其从原始文本转换为结构化字典"""
    full_text = example.get('text', '').strip()
    instruction, input_text, output = "", "", ""
    try:
        parts = full_text.split('### Assistant:')
        if len(parts) == 2:
            human_part, assistant_part = parts
            output = assistant_part.strip()
            human_content = human_part.replace('### Human:', '').strip()
            instruction_input_parts = human_content.split(',', 1)
            if len(instruction_input_parts) == 2:
                instruction = instruction_input_parts[0].strip()
                input_text = instruction_input_parts[1].strip()
    except Exception as e:
        print(f"处理数据时发生错误: {e}\n原始数据: {full_text}")
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }


def main():
    """主执行函数"""
    # 2. 加载原始数据集
    print("正在从 Hugging Face Hub 加载数据集...")
    try:
        dataset = load_dataset("HoangCuongNguyen/CTI-to-MITRE-dataset")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        sys.exit(1)  # 加载失败则退出脚本

    # 3. 选择并处理数据
    train_dataset = dataset['train']
    subset_dataset = train_dataset.select(range(NUM_SAMPLES))
    processed_dataset = subset_dataset.map(
        process_example,
        remove_columns=train_dataset.column_names
    )

    print(f"已成功处理 {len(processed_dataset)} 条数据。")

    # --- [核心修改点：手动写入 JSON 文件] ---
    # 4. 将处理后的数据收集到 Python 列表中
    print("\n正在将所有数据收集到内存列表中...")
    all_records = [record for record in processed_dataset]

    # 5. 自动创建输出目录并写入文件
    try:
        print(f"准备将文件保存到: {OUTPUT_JSON_FILE}")

        # 使用 Python 自带的 json 库进行写入
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            # json.dump 会将整个列表一次性写入，确保是单一顶层数组
            json.dump(all_records, f, indent=4, ensure_ascii=False)

        print(f"\n处理完成！数据集已通过手动方式正确保存到 {OUTPUT_JSON_FILE}")

        # 验证文件内容
        print("\n正在验证已保存的文件...")
        with open(OUTPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0, 2)  # 移动到文件末尾
            file_size = f.tell()
            f.seek(file_size - 1)
            last_char = f.read(1)
            if first_char == '[' and last_char == ']':
                print("✅ 文件验证成功：文件以 '[' 开头，以 ']' 结尾。格式正确！")
            else:
                print("❌ 文件验证失败：文件格式仍然不正确。")

    except Exception as e:
        print(f"写入文件时发生错误: {e}")


# 运行主函数
if __name__ == "__main__":
    main()