# Git Starter Guide

## 1. Setup Git (One-time)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```
## 2. Clone Repo
``` bash
git clone [your-repo](https://github.com/henreads/Capstone-AI-Chatbot.git)
cd `Capstone-AI-Chatbot`
```

## 3. Create a New Branch
```bash
git checkout main
git pull origin main                 # Make sure main is up to date
git checkout -b new-branch-name
```
## 4. Make Changes and Commit
```bash
git status                          # Check changed files
git add .                          # Stage all changes (or use 'git add filename')
git commit -m "Describe your changes"
```
## 5. Keep Your Branch Updated
```bash
git checkout main
git pull origin main                # Get latest main branch updates
git checkout your-branch-name
git merge main                     # Merge main into your branch
```

## 6. Test Your Code
```bash
Run your unit tests, linters, or build commands before pushing.
```

## 7. Push Your Branch
```bash
git push origin your-branch-name
```

## 8. Create a Pull Request (PR)
- Go to GitHub, open a PR from your branch to main
- Add a descriptive title and explanation
- Request reviews from your team

## 9. Merge and Clean Up
After PR approval and merge:
```bash
git checkout main
git pull origin main
```

# Git Quick Reference
| Action                 | Command                         |
|------------------------|--------------------------------|
| Check current branch   | `git branch`                   |
| Switch branch          | `git checkout branch-name`     |
| Create new branch      | `git checkout -b new-branch`   |
| Stage changes          | `git add .`                    |
| Commit changes         | `git commit -m "message"`      |
| Pull latest main       | `git checkout main && git pull origin main` |
| Merge main into branch | `git merge main`                |
| Push branch            | `git push origin branch-name`  |
| Delete local branch    | `git branch -d branch-name`    |

# Tips
1. Always pull latest main before creating or merging branches.
2. Write clear commit messages.
3.Test your code locally before pushing.

# Flowchart
```text
Start
  |
  v
Are there uncommitted local changes? --- Yes ---> Stage and commit changes
  |                                              git add .
  |                                              git commit -m "your message"
  |                                              v
  No                                             Merge latest main into your branch
  |                                              git checkout main
  |                                              git pull origin main
  |                                              git checkout feature/branch
  |                                              git merge main
  v                                              v
Is the merge clean and conflict-free? --- No ---> Resolve conflicts, then commit merge
  |                                              |
  Yes                                             v
  |                                          Is code working and tests passing?
  v                                              |
Is code working and tests passing? --- No ---> Fix bugs, re-test
  |                                              |
  Yes                                             v
  |                                          Ready to push changes?
  v                                              |
Ready to push changes? -------------- No ---> Continue working
  |                                              |
  Yes                                             v
  |                                          Push your branch
  v                                              git push origin feature/branch
Create Pull Request on GitHub                      |
  |                                                v
  v                                            PR review & approval
Is PR approved? --- No ---> Address review comments (repeat testing and pushing)
  |                                 |
  Yes                                v
  |                            Merge PR into main
  v                                 
 Merge PR into main
---

## Undo Steps

Did you merge by mistake or need to undo commits?

  |
  v
Is merge or commit local and NOT pushed? --- Yes ---> Use
  |                                            git reset --hard HEAD~1
  |                                            Test and fix, then push
  |
  No
  |
Is merge or commit already pushed? --- Yes ---> Use
  |                                            git revert -m 1 <merge-commit-hash>
  |                                            or git revert <commit-hash>
  |                                            Then push revert commit
  |
  No
  |
No undo needed
```

