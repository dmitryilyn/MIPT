from typing import Union


class Prompter(object):
    def generate_prompt(
        self,
        input: str,
        context: str,
        label: Union[None, str] = None,
    ) -> str:
        res = f"\n{input}\n"
        if context:
            res = f"{context}{res}"
        if label:
            res = f"{res}{label}"

        return res


    def get_response(self, output: str) -> str:
        return output.split('\n')[-1].strip()