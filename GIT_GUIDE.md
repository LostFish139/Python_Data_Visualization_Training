# Git使用指南

本指南将帮助您将重构后的项目上传回GitHub。

## 前提条件

1. 确保已安装Git
2. 确保已在GitHub上创建了远程仓库
3. 确保已配置Git用户信息：
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

## 步骤

### 1. 初始化Git仓库（如果尚未初始化）

```bash
cd d:\code\-
git init
```

### 2. 添加远程仓库（如果尚未添加）

```bash
git remote add origin https://github.com/your-username/your-repo.git
```

### 3. 查看当前状态

```bash
git status
```

### 4. 添加所有更改到暂存区

```bash
git add .
```

### 5. 提交更改

```bash
git commit -m "重构项目结构，优化代码组织"
```

### 6. 推送到GitHub

```bash
git push -u origin master
```

或者，如果您的默认分支是main：

```bash
git push -u origin main
```

## 常用Git命令

- `git status`: 查看工作目录和暂存区状态
- `git add <file>`: 将文件添加到暂存区
- `git commit -m "message"`: 提交更改
- `git push`: 将本地提交推送到远程仓库
- `git pull`: 从远程仓库拉取最新更改
- `git branch`: 查看本地分支
- `git checkout -b <branch-name>`: 创建并切换到新分支
- `git merge <branch-name>`: 合并分支

## 注意事项

1. 如果在推送时遇到冲突，请先拉取远程更改：
   ```bash
   git pull origin master
   ```
   然后解决冲突后再推送。

2. 如果您想忽略某些文件（如Python缓存文件、IDE配置等），这些文件已在.gitignore中配置。

3. 建议在每次重大更改后提交代码，并使用有意义的提交信息。

4. 如果您想删除move_to_archive.py脚本，可以在提交后执行：
   ```bash
   git rm move_to_archive.py
   git commit -m "删除临时脚本"
   git push
   ```

## 分支管理

如果您想在新分支上进行开发：

```bash
# 创建并切换到新分支
git checkout -b feature/new-feature

# 在新分支上进行开发...

# 提交更改
git add .
git commit -m "添加新功能"

# 推送到远程仓库
git push -u origin feature/new-feature
```

## 查看提交历史

```bash
git log
```

或者查看更简洁的提交历史：

```bash
git log --oneline
```
