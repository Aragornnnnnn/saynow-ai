# 배포 워크플로우의 브랜치별 인프라 매핑을 검증하는 테스트
from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _workflow_text(name: str) -> str:
    return (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def _job_environment(workflow: str) -> str:
    match = re.search(r"(?m)^        environment: (?P<environment>\S+)$", workflow)
    if match is None:
        raise AssertionError("workflow job environment is missing")
    return match.group("environment")


class DeployWorkflowTest(unittest.TestCase):
    def test_develop_branch_uses_prod_ec2_environment_but_develop_ssm_path(self):
        workflow = _workflow_text("deploy-develop.yml")

        self.assertIn("- develop", workflow)
        self.assertEqual(_job_environment(workflow), "prod")
        self.assertIn("--path /saynow/develop", workflow)
        self.assertNotIn("--path /saynow/prod", workflow)

    def test_main_branch_uses_develop_ec2_environment_but_prod_ssm_path(self):
        workflow = _workflow_text("deploy-prod.yml")

        self.assertIn("- main", workflow)
        self.assertEqual(_job_environment(workflow), "develop")
        self.assertIn("--path /saynow/prod", workflow)
        self.assertNotIn("--path /saynow/develop", workflow)

    def test_prod_workflow_accepts_raw_or_base64_ssh_key_like_develop_workflow(self):
        workflow = _workflow_text("deploy-prod.yml")

        self.assertIn("SSH_KEY: ${{ secrets.EC2_SSH_KEY }}", workflow)
        self.assertIn('grep -q "BEGIN .*PRIVATE KEY"', workflow)
        self.assertIn("base64 --decode > ~/.ssh/saynow-prod-deploy", workflow)
        self.assertIn("ssh-keygen -y -f ~/.ssh/saynow-prod-deploy > /dev/null", workflow)


if __name__ == "__main__":
    unittest.main()
