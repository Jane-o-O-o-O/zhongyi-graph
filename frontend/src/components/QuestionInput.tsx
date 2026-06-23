import { Button, Input } from 'antd';
import { Search } from 'lucide-react';
import type { FormEvent } from 'react';

type QuestionInputProps = {
  value: string;
  loading: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
};

const placeholder = '请输入中医问题，例如：失眠可以从哪些证候分析？';

export function QuestionInput({ value, loading, onChange, onSubmit }: QuestionInputProps) {
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit();
  }

  return (
    <form className="question-form" onSubmit={handleSubmit}>
      <Input
        aria-label="中医问题"
        placeholder={placeholder}
        value={value}
        disabled={loading}
        onChange={(event) => onChange(event.target.value)}
        allowClear
      />
      <Button
        htmlType="submit"
        type="primary"
        loading={loading}
        disabled={!value.trim()}
        icon={<Search size={17} />}
      >
        研判
      </Button>
    </form>
  );
}
