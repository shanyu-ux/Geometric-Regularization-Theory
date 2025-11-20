# 如果本地还没初始化 git
git init
git branch -M main

# 配置提交者信息（只需做一次）
git config user.email "ci@example.com"
git config user.name "CI Bot"

# 添加远程（替换为网页给出的 URL）
git remote add origin https://github.com/shanyu-ux/Non-Commutative-Geometric-Dynamics.git

# 提交并推送
git add .
git commit -m "Infra: Add C++ unit tests and GitHub Actions CI pipeline" || echo "No changes to commit"
git push --set-upstream origin main
