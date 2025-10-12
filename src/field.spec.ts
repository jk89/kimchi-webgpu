import { expect } from 'chai';
import { Field } from 'o1js';

describe('o1js Field.ORDER', () => {
    it('prints Field.ORDER', () => {
        const ORDER = Field.ORDER;
        console.log('Field.ORDER =', ORDER);
        expect(ORDER > 0n).to.be.true;
    });

    it('Verifies Field.add wraps modulo Field.ORDER', () => {
        const ORDER = Field.ORDER;

        const rawA = ORDER - 10n;
        const rawB = 20n;

        const expected = (rawA + rawB) % ORDER;

        const aField = Field(rawA);
        const bField = Field(rawB);
        const sumField = aField.add(bField);

        const sumFieldBigInt = sumField.toBigInt();

        console.log('rawA =', rawA);
        console.log('rawB =', rawB);
        console.log('expected (BigInt mod) =', expected);
        console.log('sum (Field) as bigint =', sumFieldBigInt);

        expect(sumFieldBigInt).to.equal(expected);
    });
});
